"""Isolated MPS warm-start retry experiment -- does NOT modify production code.

Re-tests the two untested levers called out in configs/default.yaml's
device-pinning comment after the rejected GroupNorm-only attempt: a
dampened orthogonal-init gain on MPS, and a NaN-guarded optimizer step in
the warm-start loop specifically (the main PPO update loop already has one
-- see PPO.update_batch -- but PPO.pretrain_actor does not). Also swaps MSE
for Huber loss in the warm-start objective. Uses real cached warm-start
data and train.py-matched seeding, per that same comment's verification
bar (the GroupNorm attempt was only trusted after testing this way).

This script is read-only: it loads existing warmstart_action.npy files and
writes nothing to disk, so it's safe to run alongside other training runs.

Usage
-----
python scripts/mps_warmstart_retry.py --config configs/default.yaml \
    --seeds 0 1 2 42 123 --devices mps --variants baseline experimental
python scripts/mps_warmstart_retry.py --devices cpu mps --gain 0.1
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_device
from src.env.dose_env import DoseEnv
from src.models.actor_critic import ActorCritic, ACTION_SCALE
from src.models.encoder import orthogonal_init

VARIANTS = {
    "baseline":     dict(loss_name="mse",   gain=None, guard_nan=False),
    "experimental": dict(loss_name="huber", gain=0.1,  guard_nan=True),
}


def _load_real_warmstart_data(cfg):
    """Mirrors train.py's ``_warmstart_actor`` data-collection loop."""
    env = DoseEnv(cfg, split=cfg.train_split)
    processed_split_root = Path(cfg.processed_dir) / cfg.train_split
    states, fractions, targets = [], [], []
    n_missing = 0
    for patient_id in env.patient_ids:
        warmstart_path = processed_split_root / patient_id / "warmstart_action.npy"
        if not warmstart_path.exists():
            n_missing += 1
            continue
        state, fraction_progress = env.reset(patient_id)
        warmstart_action = np.load(warmstart_path).astype(np.float32)
        states.append(torch.from_numpy(state))
        fractions.append(float(fraction_progress))
        targets.append(torch.from_numpy(warmstart_action))
    if not states:
        raise SystemExit(
            f"No warmstart_action.npy found under {processed_split_root}. "
            f"Run scripts/compute_warmstart_actions.py first."
        )
    print(f"[data] {len(states)} patient(s) ({n_missing} missing warmstart files)")
    return (torch.stack(states),
             torch.tensor(fractions, dtype=torch.float32),
             torch.stack(targets))


def _inv_softplus_targets(target_actions: torch.Tensor) -> torch.Tensor:
    """Same transform as PPO.pretrain_actor: a* -> inv_softplus(a*/scale)."""
    scaled = target_actions / float(ACTION_SCALE)
    return torch.where(
        scaled > 1e-4,
        torch.log(torch.expm1(scaled.clamp(min=1e-4))),
        torch.full_like(scaled, -8.0),
    ).clamp(min=-8.0, max=8.0)


def _dampen_gain(net: ActorCritic, gain: float) -> None:
    """Re-applies the existing ``orthogonal_init`` helper with a lower gain
    on the encoder + actor trunk (the layers pretrain_actor trains). Leaves
    actor_mu's gain=0.01 untouched -- it's already near-zero and not the
    suspected source of the gradient blow-up.
    """
    for module in net.encoder.conv:
        if isinstance(module, nn.Conv3d):
            orthogonal_init(module, gain)
    orthogonal_init(net.encoder.proj, gain)
    for module in net.actor_trunk:
        if isinstance(module, nn.Linear):
            orthogonal_init(module, gain)


def _build_loss_fn(name: str):
    if name == "mse":
        return lambda pred, target: ((pred - target) ** 2).mean()
    if name == "huber":
        huber = nn.SmoothL1Loss(beta=1.0)
        return huber
    raise ValueError(f"unknown loss {name!r}")


def _run_pretrain(net, optimizer, states_cpu, fractions_cpu, targets_raw_cpu,
                  device, *, epochs, minibatch, loss_fn, guard_nan,
                  clip_norm=1.0):
    """A copy of PPO.pretrain_actor's loop (not an edit to ppo.py) with the
    candidate loss / NaN-guard swapped in behind ``guard_nan``."""
    n_samples = states_cpu.shape[0]
    trainable_params = (list(net.encoder.parameters())
                        + list(net.actor_trunk.parameters())
                        + list(net.actor_mu.parameters()))
    history: list[float] = []
    skipped = 0
    nan_epoch = None
    diverged = False
    max_grad_norm = 0.0

    for epoch_index in range(epochs):
        shuffled = np.random.permutation(n_samples)
        running_sum, running_seen = 0.0, 0
        for batch_start in range(0, n_samples, minibatch):
            batch_idx = torch.as_tensor(
                shuffled[batch_start:batch_start + minibatch], dtype=torch.long
            )
            batch_states = states_cpu.index_select(0, batch_idx).to(device)
            batch_fp = fractions_cpu.index_select(0, batch_idx).to(device)
            batch_target = targets_raw_cpu.index_select(0, batch_idx).to(device)

            features = net.features(batch_states, batch_fp)
            predicted_mu = net.actor_mu(net.actor_trunk(features))
            loss = loss_fn(predicted_mu, batch_target)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            grad_norm_value = float(grad_norm)
            grad_finite = torch.isfinite(grad_norm)
            if grad_finite:
                max_grad_norm = max(max_grad_norm, grad_norm_value)

            if guard_nan and not grad_finite:
                skipped += 1
                if nan_epoch is None:
                    nan_epoch = epoch_index
                optimizer.zero_grad()
                continue

            optimizer.step()
            running_sum += float(loss.item()) * len(batch_idx)
            running_seen += len(batch_idx)

            if not all(torch.isfinite(p).all() for p in trainable_params):
                diverged = True
                if nan_epoch is None:
                    nan_epoch = epoch_index
                break

        history.append(running_sum / max(running_seen, 1) if running_seen
                       else float("nan"))
        if diverged:
            break

    return {"history": history, "skipped": skipped, "nan_epoch": nan_epoch,
            "max_grad_norm": max_grad_norm, "diverged": diverged}


def run_one(cfg, data, device_name, variant_name, seed, *,
            epochs, minibatch, lr, gain_override=None):
    spec = VARIANTS[variant_name]
    gain = gain_override if gain_override is not None else spec["gain"]

    device = resolve_device(device_name)
    if device_name != "cpu" and device.type == "cpu":
        return {"skipped_device": True}

    np.random.seed(seed)
    torch.manual_seed(seed)

    states_cpu, fractions_cpu, targets_raw_cpu = data
    in_channels = states_cpu.shape[1]
    n_beamlets = targets_raw_cpu.shape[1]

    net = ActorCritic(
        in_channels, n_beamlets,
        log_std_init=float(getattr(cfg, "actor_log_std_init", -1.0)),
        log_std_max=float(getattr(cfg, "actor_log_std_max", 2.0)),
    ).to(device)
    if gain is not None:
        _dampen_gain(net, gain)

    trainable_params = (list(net.encoder.parameters())
                        + list(net.actor_trunk.parameters())
                        + list(net.actor_mu.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = _build_loss_fn(spec["loss_name"])

    result = _run_pretrain(
        net, optimizer, states_cpu, fractions_cpu, targets_raw_cpu, device,
        epochs=epochs, minibatch=minibatch, loss_fn=loss_fn,
        guard_nan=spec["guard_nan"],
    )
    result.update(device=str(device), variant=variant_name, seed=seed,
                 gain=gain, skipped_device=False)
    return result


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--seeds", type=int, nargs="+",
                            default=[0, 1, 2, 42, 123])
    arg_parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    arg_parser.add_argument("--variants", nargs="+",
                            default=["baseline", "experimental"],
                            choices=list(VARIANTS))
    arg_parser.add_argument("--epochs", type=int, default=None,
                            help="default: cfg.warmstart_epochs")
    arg_parser.add_argument("--minibatch", type=int, default=None,
                            help="default: cfg.warmstart_minibatch")
    arg_parser.add_argument("--lr", type=float, default=None,
                            help="default: cfg.warmstart_lr")
    arg_parser.add_argument("--gain", type=float, default=None,
                            help="override the experimental variant's "
                                 "dampened init gain (default 0.1)")
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs or cfg.warmstart_epochs
    minibatch = args.minibatch or cfg.warmstart_minibatch
    lr = args.lr or cfg.warmstart_lr

    print("[data] loading real cached warm-start actions...")
    states_cpu, fractions_cpu, targets_cpu = _load_real_warmstart_data(cfg)
    targets_raw_cpu = _inv_softplus_targets(targets_cpu)
    data = (states_cpu, fractions_cpu, targets_raw_cpu)

    rows = []
    for variant_name in args.variants:
        for seed in args.seeds:
            for device_name in args.devices:
                print(f"\n[run] variant={variant_name} seed={seed} "
                      f"device={device_name}")
                result = run_one(
                    cfg, data, device_name, variant_name, seed,
                    epochs=epochs, minibatch=minibatch, lr=lr,
                    gain_override=args.gain,
                )
                if result.get("skipped_device"):
                    print(f"  [skip] '{device_name}' not available "
                          f"on this machine")
                    continue
                history = result["history"]
                print(f"  diverged={result['diverged']}  "
                      f"nan_epoch={result['nan_epoch']}  "
                      f"skipped_steps={result['skipped']}  "
                      f"max_grad_norm={result['max_grad_norm']:.3f}")
                print(f"  loss: first={history[0]:.4f}  last={history[-1]:.4f}")
                rows.append(result)

    print("\n[summary] variant       device  seed  diverged  nan_epoch  "
          "skipped  first_loss  last_loss")
    for r in rows:
        print(f"  {r['variant']:<12} {r['device']:<6} {r['seed']:<5} "
              f"{str(r['diverged']):<9} {str(r['nan_epoch']):<10} "
              f"{r['skipped']:<8} {r['history'][0]:<11.4f} "
              f"{r['history'][-1]:.4f}")


if __name__ == "__main__":
    main()
