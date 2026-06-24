"""Benchmark CPU vs GPU (CUDA/MPS) for the PPO training pipeline.

Builds synthetic data with the exact tensor shapes ``train.py`` produces
(``cfg.grid``, ``cfg.n_beamlets``, ``cfg.n_fractions``, ``cfg.batch_n_patients``)
so timings reflect the real per-episode cost without needing preprocessed
OpenKBP patients. Two stages are timed separately because they stress the
hardware differently:

  * ``act``    — ``cfg.n_fractions`` sequential single-sample CNN forward
                 passes (this is how ``collect_patient`` gathers a rollout;
                 it is latency-, not throughput-, bound).
  * ``update`` — one ``PPO.update_batch`` call over a full batch of patient
                 trajectories (large batched forward/backward passes;
                 throughput-bound).

Usage
-----
python scripts/benchmark_device.py --config configs/default.yaml
python scripts/benchmark_device.py --devices cpu mps --n-episodes 3
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_device, ALL_STRUCTURES
from src.agents.ppo import PPO, Rollout


def _make_synthetic_rollout(cfg, in_channels: int) -> Rollout:
    """One patient course of random data with the real training shapes."""
    n_fractions = cfg.n_fractions
    states = torch.rand(n_fractions, in_channels, cfg.grid, cfg.grid, cfg.grid)
    fraction_progresses = torch.arange(n_fractions, dtype=torch.float32) / n_fractions
    raw_action_samples = torch.randn(n_fractions, cfg.n_beamlets)
    log_probabilities = torch.randn(n_fractions)
    rewards = torch.randn(n_fractions)
    values = torch.randn(n_fractions)
    dones = torch.zeros(n_fractions)
    dones[-1] = 1.0
    return Rollout(states, fraction_progresses, raw_action_samples,
                   log_probabilities, rewards, values, dones)


def _bench_act(agent: PPO, cfg, in_channels: int, n_fractions: int) -> float:
    state = np.random.rand(in_channels, cfg.grid, cfg.grid, cfg.grid).astype(np.float32)
    start = time.perf_counter()
    for fraction_index in range(n_fractions):
        agent.act(state, fraction_index / n_fractions)
    _sync(agent.device)
    return time.perf_counter() - start


def _bench_update(agent: PPO, rollouts: list[Rollout]) -> float:
    start = time.perf_counter()
    agent.update_batch(rollouts, last_value=0.0)
    _sync(agent.device)
    return time.perf_counter() - start


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_benchmark(cfg, device_name: str, n_episodes: int) -> dict:
    cfg.device = device_name
    in_channels = 1 + len(ALL_STRUCTURES) + 1 + 1 + 1  # CT + masks + dose + gap + beams
    agent = PPO(cfg, in_channels=in_channels)
    resolved = agent.device
    if device_name != "cpu" and resolved.type == "cpu":
        print(f"  [skip] '{device_name}' not available on this machine")
        return {}

    batch_n_patients = max(1, int(getattr(cfg, "batch_n_patients", 8)))

    # warm-up (compiles kernels / allocates buffers; excluded from timing)
    _bench_act(agent, cfg, in_channels, n_fractions=2)
    _bench_update(agent, [_make_synthetic_rollout(cfg, in_channels)])

    act_times, update_times = [], []
    for _ in range(n_episodes):
        act_times.append(
            _bench_act(agent, cfg, in_channels, n_fractions=cfg.n_fractions)
        )
        rollouts = [_make_synthetic_rollout(cfg, in_channels)
                   for _ in range(batch_n_patients)]
        update_times.append(_bench_update(agent, rollouts))

    return {
        "device": str(resolved),
        "act_s_per_patient": float(np.mean(act_times)),
        "update_s_per_batch": float(np.mean(update_times)),
        "update_s_per_patient": float(np.mean(update_times)) / batch_n_patients,
        "total_s_per_patient": (float(np.mean(act_times))
                                + float(np.mean(update_times)) / batch_n_patients),
    }


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--devices", nargs="+", default=["cpu", "mps", "cuda"])
    arg_parser.add_argument("--n-episodes", type=int, default=5,
                            help="timed repetitions per device (after warm-up)")
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)

    print(f"[benchmark] grid={cfg.grid}^3  n_beamlets={cfg.n_beamlets}  "
          f"n_fractions={cfg.n_fractions}  batch_n_patients={cfg.batch_n_patients}  "
          f"ppo_epochs={cfg.ppo_epochs}  minibatch={cfg.minibatch}")

    results = {}
    for device_name in args.devices:
        print(f"\n[benchmark] device = {device_name}")
        result = run_benchmark(cfg, device_name, args.n_episodes)
        if result:
            results[device_name] = result
            print(f"  resolved device        : {result['device']}")
            print(f"  act() / patient (35 fx) : {result['act_s_per_patient']:.3f} s")
            print(f"  update_batch() / batch  : {result['update_s_per_batch']:.3f} s")
            print(f"  update_batch() / patient: {result['update_s_per_patient']:.3f} s")
            print(f"  TOTAL / patient-episode : {result['total_s_per_patient']:.3f} s")

    if len(results) > 1:
        print("\n[benchmark] summary (seconds per patient-episode)")
        baseline_name, baseline = next(iter(results.items()))
        for device_name, result in results.items():
            speedup = baseline["total_s_per_patient"] / result["total_s_per_patient"]
            print(f"  {device_name:>5}: {result['total_s_per_patient']:.3f} s  "
                  f"({speedup:.2f}x vs {baseline_name})")


if __name__ == "__main__":
    main()
