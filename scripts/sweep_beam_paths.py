"""Beam-paths channel ablation, compared by validation DVH score.

Why no shared warm-start (unlike scripts/sweep_gamma.py): gamma doesn't
change the network's input shape, so that sweep warm-starts the actor once
and shares the checkpoint across every gamma candidate. The beam_paths
toggle is different -- it changes ``in_channels`` itself (12 with the
channel, 11 without, see ``Config.include_beam_paths`` /
``DoseEnv._build_state``), which changes the encoder's first ``Conv3d``
shape. ``PPO.load``'s ``load_state_dict`` call is unguarded and will
hard-crash on that shape mismatch, so the two variants' checkpoints are
never interchangeable. Each candidate here runs its own independent
warm-start + PPO loop from scratch, in its own ``ckpt_dir``.

Validation DVH score (``train._validation_dvh_score``, the same metric the
project already uses for ``best.pt`` selection) is the comparison axis --
it's gamma-/config-independent and is simply this project's standard
model-quality metric.

Approach: for each variant ("on" -> ``include_beam_paths: true``, "off" ->
``include_beam_paths: false``), write a candidate config (same ``seed`` as
the base config, so both variants sample patients in the same order) and
run ``train.py --config <candidate>/config.yaml --episodes <n>`` with no
``--resume`` -- this reuses ``train.py`` end-to-end (warm-start, curriculum,
batching, logging) rather than duplicating its training loop.

Usage
-----
python scripts/sweep_beam_paths.py --episodes 300
python scripts/sweep_beam_paths.py --episodes 26 --sweep-root runs/sweep_beam_paths_smoke   # smoke test
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

VARIANT_FLAGS = {"on": True, "off": False}


def _run_candidate(base_config: dict, variant: str, episodes: int,
                   sweep_root: Path) -> Path:
    candidate_dir = sweep_root / f"beam_paths_{variant}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = dict(base_config)
    candidate_config["include_beam_paths"] = VARIANT_FLAGS[variant]
    candidate_config["ckpt_dir"] = str(candidate_dir)
    candidate_config_path = candidate_dir / "config.yaml"
    candidate_config_path.write_text(yaml.safe_dump(candidate_config))

    log_path = candidate_dir / "subprocess.log"
    print(f"[sweep_beam_paths] beam_paths={variant}: running {episodes} "
          f"episodes -> {candidate_dir} (log: {log_path})")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            [sys.executable, "train.py",
             "--config", str(candidate_config_path),
             "--episodes", str(episodes)],
            cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[sweep_beam_paths] beam_paths={variant}: {status}")
    return candidate_dir / "train_log.csv"


def _summarize(results: dict[str, Path], sweep_root: Path) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, log_path in results.items():
        if not log_path.is_file():
            rows.append({"variant": variant, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "missing log"})
            continue
        df = pd.read_csv(log_path)
        val_df = df.dropna(subset=["val_dvh"])
        if val_df.empty:
            rows.append({"variant": variant, "best_val_dvh": float("nan"),
                        "final_val_dvh": float("nan"), "status": "no validation rows"})
            continue
        best_val_dvh = float(val_df["val_dvh"].min())
        final_val_dvh = float(val_df["val_dvh"].iloc[-1])
        rows.append({"variant": variant, "best_val_dvh": best_val_dvh,
                    "final_val_dvh": final_val_dvh, "status": "ok"})
        ax.plot(val_df["episode"], val_df["val_dvh"], "o-", label=variant)

    ax.set_xlabel("episode")
    ax.set_ylabel("validation DVH score (lower = better)")
    ax.set_title("beam_paths ablation -- validation DVH vs episode")
    ax.legend(loc="best")
    fig.tight_layout()
    comparison_png = sweep_root / "comparison.png"
    fig.savefig(comparison_png, dpi=120)
    plt.close(fig)

    summary_df = pd.DataFrame(rows).sort_values("best_val_dvh", na_position="last")
    summary_csv = sweep_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n[sweep_beam_paths] summary (sorted by best validation DVH score, lower = better)")
    print(summary_df.to_string(index=False))
    print(f"\n[sweep_beam_paths] saved {summary_csv}")
    print(f"[sweep_beam_paths] saved {comparison_png}")
    if (summary_df["status"] == "ok").any():
        winner = summary_df.iloc[0]
        print(f"\n[sweep_beam_paths] best candidate: beam_paths={winner['variant']} "
              f"(best_val_dvh={winner['best_val_dvh']:.3f})")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--variants", nargs="+", choices=["on", "off"],
                            default=["on", "off"])
    arg_parser.add_argument("--episodes", type=int, default=300,
                            help="episodes per candidate (default 300)")
    arg_parser.add_argument("--sweep-root", default="runs/sweep_beam_paths")
    args = arg_parser.parse_args()

    config_path = Path(args.config)
    base_config = yaml.safe_load(config_path.read_text())
    sweep_root = Path(args.sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for variant in args.variants:
        results[variant] = _run_candidate(
            base_config, variant, args.episodes, sweep_root,
        )

    _summarize(results, sweep_root)


if __name__ == "__main__":
    main()
