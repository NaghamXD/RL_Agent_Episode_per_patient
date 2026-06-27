"""Plot a ``train_log.csv`` produced by ``src.utils.train_logger.TrainingLogger``.

This is what actually answers "is total_episodes enough?" -- the project
has no other surviving record of a training curve (``runs/`` is
gitignored, and the only file checked in is a single pasted-in final
evaluation). Look at the bottom (validation DVH) panel: if it's still
trending down near the end of the run, training likely helps from more
episodes; if it has flattened well before the end, the run could be
shortened for faster iteration.

Usage
-----
python scripts/plot_training_curve.py --log runs/train_log.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(log_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(log_path)
    val_df = df.dropna(subset=["val_dvh"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(df["episode"], df["patient_reward"], alpha=0.3, label="reward (raw)")
    axes[0].plot(df["episode"], df["rolling_mean_reward"], label="reward (rolling mean)")
    axes[0].set_ylabel("training reward")
    axes[0].legend(loc="best")
    axes[0].set_title(f"training curve -- {log_path}")

    # entropy is summed over all action dimensions (thousands of beamlets),
    # so it's orders of magnitude larger than the losses -- give it its own
    # axis rather than letting it flatten policy/value loss to a line at 0.
    loss_lines = axes[1].plot(df["episode"], df["policy_loss"], label="policy loss", color="tab:blue")
    loss_lines += axes[1].plot(df["episode"], df["value_loss"], label="value loss", color="tab:orange")
    axes[1].set_ylabel("PPO losses (per-update mean)")

    entropy_axis = axes[1].twinx()
    entropy_lines = entropy_axis.plot(df["episode"], df["entropy"], label="entropy", color="tab:green")
    entropy_axis.set_ylabel("policy entropy (summed over action dims)")

    axes[1].legend(loss_lines + entropy_lines, [line.get_label() for line in loss_lines + entropy_lines], loc="best")

    if len(val_df):
        axes[2].plot(val_df["episode"], val_df["val_dvh"], "o-",
                    label="val DVH score (lower = better)")
        best_row = val_df.loc[val_df["val_dvh"].idxmin()]
        axes[2].scatter([best_row["episode"]], [best_row["val_dvh"]],
                        color="red", zorder=5, label="best")
    axes[2].set_ylabel("validation DVH score")
    axes[2].set_xlabel("episode")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log", default="runs/train_log.csv")
    arg_parser.add_argument(
        "--out", default=None,
        help="output PNG path (defaults to <log> with a .png suffix)",
    )
    args = arg_parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"log file not found: {log_path}")
    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")

    saved = plot_training_curve(log_path, out_path)
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
