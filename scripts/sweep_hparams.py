"""Hyperparameter regime sweep, compared by validation DVH score.

Four named regimes bundle five fields each (batch_n_patients, minibatch,
ppo_epochs, lr, lambda_oar_ramp_episodes) -- each row is a coarse "preset"
to compare as a whole, not a one-variable-at-a-time ablation. ``baseline``
is the real ``configs/default.yaml`` (the config that actually produced
``runs/train_log.csv``'s 3500-episode run), not the ``src/config.py``
dataclass defaults.

None of these five fields touch network architecture (``in_channels``,
``n_beamlets``, ``latent_dim``, ``hidden_dim``, ``log_std_*`` are the only
``ActorCritic.__init__`` params and none derive from these hyperparameters),
so -- unlike the beam_paths channel ablation -- warm-start CAN be shared
across all four regimes, exactly like ``scripts/sweep_gamma.py``.

``baseline`` is special-cased to reuse the existing ``runs/train_log.csv``
instead of re-running 3500 episodes: its ``val_dvh`` column comes from
``train._validation_dvh_score``, a deterministic function unrelated to the
per-update-mean logging fix applied to ``policy_loss``/``value_loss``/
``entropy``, so it remains valid for comparison. Pass ``--rerun-baseline``
to run a fresh baseline candidate instead.

Usage
-----
python scripts/sweep_hparams.py
python scripts/sweep_hparams.py --regimes sweep1_minibatch_active --episodes 20   # smoke test
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Bundled hyperparameter overrides per named regime. ``baseline`` matches
# configs/default.yaml exactly (the config that produced runs/train_log.csv).
REGIMES: dict[str, dict] = {
    "baseline": {
        "batch_n_patients": 8, "minibatch": 64, "ppo_epochs": 6,
        "lr": 1.5e-4, "lambda_oar_ramp_episodes": 600,
    },
    "sweep1_minibatch_active": {
        "batch_n_patients": 4, "minibatch": 64, "ppo_epochs": 3,
        "lr": 1e-4, "lambda_oar_ramp_episodes": 200,
    },
    "sweep2_heavy_reg": {
        "batch_n_patients": 8, "minibatch": 128, "ppo_epochs": 3,
        "lr": 5e-5, "lambda_oar_ramp_episodes": 300,
    },
    "sweep3_aggressive_adaptivity": {
        "batch_n_patients": 2, "minibatch": 32, "ppo_epochs": 4,
        "lr": 1e-4, "lambda_oar_ramp_episodes": 200,
    },
}

HISTORICAL_TRAIN_LOG = PROJECT_ROOT / "runs" / "train_log.csv"


def _ensure_warmstart(config_path: Path, warmstart_ckpt: Path, force: bool) -> None:
    if warmstart_ckpt.is_file() and not force:
        print(f"[sweep_hparams] reusing existing warm-start checkpoint {warmstart_ckpt}")
        return
    print(f"[sweep_hparams] no warm-start checkpoint at {warmstart_ckpt}; "
          f"running train.py --episodes 0 to create one (shared across all regimes)")
    subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path), "--episodes", "0"],
        cwd=PROJECT_ROOT, check=True,
    )
    if not warmstart_ckpt.is_file():
        raise SystemExit(
            f"[sweep_hparams] expected {warmstart_ckpt} after warm-start run; "
            f"check cfg.ckpt_dir in {config_path} and cfg.warmstart_enabled."
        )


def _reuse_historical_baseline(sweep_root: Path) -> Path:
    candidate_dir = sweep_root / "baseline"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if not HISTORICAL_TRAIN_LOG.is_file():
        raise SystemExit(
            f"[sweep_hparams] --rerun-baseline not given but "
            f"{HISTORICAL_TRAIN_LOG} doesn't exist; pass --rerun-baseline to "
            f"train a fresh baseline candidate instead."
        )
    dest = candidate_dir / "train_log.csv"
    shutil.copy2(HISTORICAL_TRAIN_LOG, dest)
    print(f"[sweep_hparams] baseline: reused historical {HISTORICAL_TRAIN_LOG} -> {dest}")
    for ckpt_name in ("last.pt", "best.pt"):
        src_ckpt = HISTORICAL_TRAIN_LOG.parent / ckpt_name
        if src_ckpt.is_file():
            shutil.copy2(src_ckpt, candidate_dir / ckpt_name)
    return dest


def _run_candidate(base_config: dict, regime: str, episodes: int,
                   warmstart_ckpt: Path, sweep_root: Path) -> Path:
    candidate_dir = sweep_root / regime
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = dict(base_config)
    candidate_config.update(REGIMES[regime])
    candidate_config["ckpt_dir"] = str(candidate_dir)
    candidate_config_path = candidate_dir / "config.yaml"
    candidate_config_path.write_text(yaml.safe_dump(candidate_config))

    log_path = candidate_dir / "subprocess.log"
    print(f"[sweep_hparams] {regime}: running {episodes} episodes "
          f"-> {candidate_dir} (log: {log_path})")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            [sys.executable, "train.py",
             "--config", str(candidate_config_path),
             "--resume", str(warmstart_ckpt),
             "--episodes", str(episodes)],
            cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[sweep_hparams] {regime}: {status}")
    return candidate_dir / "train_log.csv"


def _summarize(results: dict[str, Path], sweep_root: Path) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for regime, log_path in results.items():
        if not log_path.is_file():
            rows.append({"regime": regime, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "missing log"})
            continue
        df = pd.read_csv(log_path)
        val_df = df.dropna(subset=["val_dvh"])
        if val_df.empty:
            rows.append({"regime": regime, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "no validation rows"})
            continue
        best_val_dvh = float(val_df["val_dvh"].min())
        final_val_dvh = float(val_df["val_dvh"].iloc[-1])
        rows.append({"regime": regime, "best_val_dvh": best_val_dvh,
                    "final_val_dvh": final_val_dvh, "status": "ok"})
        ax.plot(val_df["episode"], val_df["val_dvh"], "o-", label=regime)

    ax.set_xlabel("episode")
    ax.set_ylabel("validation DVH score (lower = better)")
    ax.set_title("hyperparameter regime sweep -- validation DVH vs episode")
    ax.legend(loc="best")
    fig.tight_layout()
    comparison_png = sweep_root / "comparison.png"
    fig.savefig(comparison_png, dpi=120)
    plt.close(fig)

    summary_df = pd.DataFrame(rows).sort_values("best_val_dvh", na_position="last")
    summary_csv = sweep_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n[sweep_hparams] summary (sorted by best validation DVH score, lower = better)")
    print(summary_df.to_string(index=False))
    print(f"\n[sweep_hparams] saved {summary_csv}")
    print(f"[sweep_hparams] saved {comparison_png}")
    if (summary_df["status"] == "ok").any():
        winner = summary_df.iloc[0]
        print(f"\n[sweep_hparams] best regime: {winner['regime']} "
              f"(best_val_dvh={winner['best_val_dvh']:.3f})")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--regimes", nargs="+", choices=list(REGIMES.keys()),
                            default=list(REGIMES.keys()))
    arg_parser.add_argument("--episodes", type=int, default=600,
                            help="episodes per non-baseline regime (default 600)")
    arg_parser.add_argument("--rerun-baseline", action="store_true",
                            help="train a fresh baseline candidate instead of "
                                 "reusing runs/train_log.csv")
    arg_parser.add_argument("--sweep-root", default="runs/sweep_hparams")
    arg_parser.add_argument("--warmstart-ckpt", default=None,
                            help="defaults to <cfg.ckpt_dir>/warmstart.pt")
    arg_parser.add_argument("--force-warmstart", action="store_true",
                            help="recompute the shared warm-start checkpoint "
                                 "even if it already exists")
    args = arg_parser.parse_args()

    config_path = Path(args.config)
    base_config = yaml.safe_load(config_path.read_text())
    warmstart_ckpt = (Path(args.warmstart_ckpt) if args.warmstart_ckpt
                      else Path(base_config.get("ckpt_dir", "runs")) / "warmstart.pt")
    sweep_root = Path(args.sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    _ensure_warmstart(config_path, warmstart_ckpt, force=args.force_warmstart)

    results: dict[str, Path] = {}
    for regime in args.regimes:
        if regime == "baseline" and not args.rerun_baseline:
            results[regime] = _reuse_historical_baseline(sweep_root)
        else:
            results[regime] = _run_candidate(
                base_config, regime, args.episodes, warmstart_ckpt, sweep_root,
            )

    _summarize(results, sweep_root)


if __name__ == "__main__":
    main()
