"""Coarse gamma sweep, compared by validation DVH score.

Why validation DVH and not training reward: in sequential mode,
``DoseEnv.step`` bakes ``cfg.gamma`` directly into the potential-based
shaping term (``reward = gamma * phi(s') - phi(s)``, see
``src/env/reward.py``/``src/env/dose_env.py``), and ``PPO._compute_gae``
uses the same ``cfg.gamma`` for bootstrapping. Both must match for the
shaping to stay policy-invariant, which also means **changing gamma
changes the reward landscape itself** -- raw training reward across
different gamma values is not comparable. Validation DVH score
(``train._validation_dvh_score``, the same metric the project already
uses for ``best.pt`` selection) is gamma-independent and is the right
axis for comparison.

Approach: warm-start the actor *once* (shared across all candidates,
since the NNLS warm-start target doesn't depend on gamma), then branch a
short PPO run per candidate from that shared starting point via
``train.py --resume <warmstart.pt> --episodes <n>`` -- this reuses
``train.py`` end-to-end (curriculum, batching, logging) rather than
duplicating its training loop.

Usage
-----
python scripts/sweep_gamma.py --episodes 900
python scripts/sweep_gamma.py --gammas 0.97 0.99 0.999 --episodes 200   # smoke test
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_warmstart(config_path: Path, warmstart_ckpt: Path, force: bool) -> None:
    if warmstart_ckpt.is_file() and not force:
        print(f"[sweep_gamma] reusing existing warm-start checkpoint {warmstart_ckpt}")
        return
    print(f"[sweep_gamma] no warm-start checkpoint at {warmstart_ckpt}; "
          f"running train.py --episodes 0 to create one (shared across all candidates)")
    subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path), "--episodes", "0"],
        cwd=PROJECT_ROOT, check=True,
    )
    if not warmstart_ckpt.is_file():
        raise SystemExit(
            f"[sweep_gamma] expected {warmstart_ckpt} after warm-start run; "
            f"check cfg.ckpt_dir in {config_path} and cfg.warmstart_enabled."
        )


def _run_candidate(base_config: dict, gamma: float, episodes: int,
                   warmstart_ckpt: Path, sweep_root: Path) -> Path:
    candidate_dir = sweep_root / f"g_{gamma}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = dict(base_config)
    candidate_config["gamma"] = float(gamma)
    candidate_config["ckpt_dir"] = str(candidate_dir)
    candidate_config_path = candidate_dir / "config.yaml"
    candidate_config_path.write_text(yaml.safe_dump(candidate_config))

    log_path = candidate_dir / "subprocess.log"
    print(f"[sweep_gamma] gamma={gamma}: running {episodes} episodes "
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
    print(f"[sweep_gamma] gamma={gamma}: {status}")
    return candidate_dir / "train_log.csv"


def _summarize(results: dict[float, Path], sweep_root: Path) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma, log_path in results.items():
        if not log_path.is_file():
            rows.append({"gamma": gamma, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "missing log"})
            continue
        df = pd.read_csv(log_path)
        val_df = df.dropna(subset=["val_dvh"])
        if val_df.empty:
            rows.append({"gamma": gamma, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "no validation rows"})
            continue
        best_val_dvh = float(val_df["val_dvh"].min())
        final_val_dvh = float(val_df["val_dvh"].iloc[-1])
        rows.append({"gamma": gamma, "best_val_dvh": best_val_dvh,
                    "final_val_dvh": final_val_dvh, "status": "ok"})
        ax.plot(val_df["episode"], val_df["val_dvh"], "o-", label=f"gamma={gamma}")

    ax.set_xlabel("episode")
    ax.set_ylabel("validation DVH score (lower = better)")
    ax.set_title("gamma sweep -- validation DVH vs episode")
    ax.legend(loc="best")
    fig.tight_layout()
    comparison_png = sweep_root / "comparison.png"
    fig.savefig(comparison_png, dpi=120)
    plt.close(fig)

    summary_df = pd.DataFrame(rows).sort_values("best_val_dvh", na_position="last")
    summary_csv = sweep_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n[sweep_gamma] summary (sorted by best validation DVH score, lower = better)")
    print(summary_df.to_string(index=False))
    print(f"\n[sweep_gamma] saved {summary_csv}")
    print(f"[sweep_gamma] saved {comparison_png}")
    if (summary_df["status"] == "ok").any():
        winner = summary_df.iloc[0]
        print(f"\n[sweep_gamma] best candidate: gamma={winner['gamma']} "
              f"(best_val_dvh={winner['best_val_dvh']:.3f})")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--gammas", nargs="+", type=float,
                            default=[0.95, 0.97, 0.99, 0.995, 0.999])
    arg_parser.add_argument("--episodes", type=int, default=900,
                            help="episodes per candidate (enough to clear "
                                 "lambda_oar_ramp_episodes, default 900)")
    arg_parser.add_argument("--sweep-root", default="runs/sweep_gamma")
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

    results: dict[float, Path] = {}
    for gamma in args.gammas:
        results[gamma] = _run_candidate(
            base_config, gamma, args.episodes, warmstart_ckpt, sweep_root,
        )

    _summarize(results, sweep_root)


if __name__ == "__main__":
    main()
