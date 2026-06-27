"""Compare the 3 reward-shaping improvements (src/env/reward.py) in isolation.

Background: the 4-regime clinical-metrics eval (D95 per PTV, mean dose per
OAR -- see ``evaluate.py``) showed every regime substantially underdoses
every PTV (D95 ~38-44 Gy vs Rx 56-70 Gy) and overdoses both parotids past
their 26 Gy tolerance by 1.4-1.9x. Three changes to the shaping potential
``phi`` and the terminal objective were proposed to address this:

  C. ``terminal_use_soft_coverage``: sigmoid terminal coverage instead of
     a hard ``dose >= Rx`` step (differentiable critic target).
  A. ``ptv_gap_power=2.0``: square the PTV gap in ``phi`` instead of linear
     (steeper gradient for severely underdosed voxels). Shrinks phi's
     magnitude, so a `lambda_phi` retune is tested alongside it.
  B. ``oar_barrier_steepness``: soft exponential OAR-overshoot barrier in
     ``phi`` instead of a hard threshold (anticipatory pressure before an
     organ crosses tolerance, not just after).

All three are off by default (see ``src/config.py`` / ``configs/default.yaml``)
-- this script is what actually turns them on, one at a time, layered in
the agreed C -> A -> B order, so it's clear which change moved which
clinical metric. Mirrors ``scripts/sweep_gamma.py``'s pattern: shared
warm-start once, branch a PPO run per variant via
``train.py --resume <warmstart.pt> --episodes <n>``, then run
``evaluate.py`` against each variant's ``best.pt`` for the clinical-metrics
table.

Usage
-----
python scripts/sweep_reward_shaping.py --episodes 2000
python scripts/sweep_reward_shaping.py --variants control C --episodes 200   # smoke test
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Layered in the agreed C -> A -> B order. ``lambda_phi=None`` means "leave
# the base config's value alone". C_A_retuned's lambda_phi=2.0 is a first
# guess at compensating A's magnitude shrinkage (gap=0.4 -> 0.16, roughly
# 2.5x smaller at that gap level) -- not a derived value, that's exactly
# what this sweep is for.
VARIANTS: dict[str, dict] = {
    "control":     dict(terminal_use_soft_coverage=False, ptv_gap_power=1.0,
                        oar_barrier_steepness=None, lambda_phi=None),
    "C":           dict(terminal_use_soft_coverage=True,  ptv_gap_power=1.0,
                        oar_barrier_steepness=None, lambda_phi=None),
    "C_A":         dict(terminal_use_soft_coverage=True,  ptv_gap_power=2.0,
                        oar_barrier_steepness=None, lambda_phi=None),
    "C_A_retuned": dict(terminal_use_soft_coverage=True,  ptv_gap_power=2.0,
                        oar_barrier_steepness=None, lambda_phi=2.0),
    "C_A_B":       dict(terminal_use_soft_coverage=True,  ptv_gap_power=2.0,
                        oar_barrier_steepness=2.0,  lambda_phi=2.0),
}


def _ensure_warmstart(config_path: Path, warmstart_ckpt: Path, force: bool) -> None:
    if warmstart_ckpt.is_file() and not force:
        print(f"[sweep_reward_shaping] reusing existing warm-start checkpoint {warmstart_ckpt}")
        return
    print(f"[sweep_reward_shaping] no warm-start checkpoint at {warmstart_ckpt}; "
          f"running train.py --episodes 0 to create one (shared across all variants)")
    subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path), "--episodes", "0"],
        cwd=PROJECT_ROOT, check=True,
    )
    if not warmstart_ckpt.is_file():
        raise SystemExit(
            f"[sweep_reward_shaping] expected {warmstart_ckpt} after warm-start run; "
            f"check cfg.ckpt_dir in {config_path} and cfg.warmstart_enabled."
        )


def _run_candidate(base_config: dict, variant_name: str, overrides: dict,
                   episodes: int, warmstart_ckpt: Path, sweep_root: Path,
                   device: str) -> Path:
    candidate_dir = sweep_root / variant_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = dict(base_config)
    candidate_config["device"] = device
    candidate_config["ckpt_dir"] = str(candidate_dir)
    for key, value in overrides.items():
        if value is None and key == "lambda_phi":
            continue  # "leave base config's lambda_phi alone"
        candidate_config[key] = value
    candidate_config_path = candidate_dir / "config.yaml"
    candidate_config_path.write_text(yaml.safe_dump(candidate_config))

    log_path = candidate_dir / "subprocess.log"
    print(f"[sweep_reward_shaping] {variant_name}: running {episodes} episodes "
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
    print(f"[sweep_reward_shaping] {variant_name}: train {status}")
    return candidate_dir


_MEAN_ROW_RE = re.compile(r"^MEAN\s*\|(.+)$")
_HEADER_ROW_RE = re.compile(r"^Patient\s*\|(.+)$")


def _evaluate_candidate(candidate_dir: Path, eval_split: str | None) -> dict:
    """Run evaluate.py against this variant's best.pt and parse its MEAN row."""
    config_path = candidate_dir / "config.yaml"
    ckpt_path = candidate_dir / "best.pt"
    if not ckpt_path.is_file():
        return {"status": "no best.pt"}

    eval_log_path = candidate_dir / "eval_results.log"
    cmd = [sys.executable, "evaluate.py", "--config", str(config_path),
          "--ckpt", str(ckpt_path)]
    if eval_split:
        cmd += ["--split", eval_split]
    with open(eval_log_path, "w") as log_file:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_file,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {"status": f"evaluate.py FAILED (exit {result.returncode})"}

    log_text = eval_log_path.read_text()
    header_match = None
    mean_match = None
    for line in log_text.splitlines():
        if header_match is None:
            header_match = _HEADER_ROW_RE.match(line.strip())
        m = _MEAN_ROW_RE.match(line.strip())
        if m:
            mean_match = m
    if header_match is None or mean_match is None:
        return {"status": "could not parse evaluate.py output"}

    headers = [h.strip() for h in header_match.group(1).split("|")]
    values = [v.strip() for v in mean_match.group(1).split("|")]
    metrics = {"status": "ok"}
    for header, value in zip(headers, values):
        try:
            metrics[header] = float(value)
        except ValueError:
            metrics[header] = float("nan")
    return metrics


def _summarize(variant_dirs: dict[str, Path], sweep_root: Path,
               eval_split: str | None) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_name, candidate_dir in variant_dirs.items():
        log_path = candidate_dir / "train_log.csv"
        row = {"variant": variant_name}
        if not log_path.is_file():
            row.update(best_val_dvh=float("nan"), final_val_dvh=float("nan"),
                      status="missing train_log.csv")
            rows.append(row)
            continue
        df = pd.read_csv(log_path)
        val_df = df.dropna(subset=["val_dvh"])
        if val_df.empty:
            row.update(best_val_dvh=float("nan"), final_val_dvh=float("nan"),
                      status="no validation rows")
            rows.append(row)
            continue
        row["best_val_dvh"] = float(val_df["val_dvh"].min())
        row["final_val_dvh"] = float(val_df["val_dvh"].iloc[-1])
        ax.plot(val_df["episode"], val_df["val_dvh"], "o-", label=variant_name)

        clinical = _evaluate_candidate(candidate_dir, eval_split)
        row.update(clinical)
        rows.append(row)

    ax.set_xlabel("episode")
    ax.set_ylabel("validation DVH score (lower = better)")
    ax.set_title("reward-shaping sweep -- validation DVH vs episode")
    ax.legend(loc="best")
    fig.tight_layout()
    comparison_png = sweep_root / "comparison.png"
    fig.savefig(comparison_png, dpi=120)
    plt.close(fig)

    summary_df = pd.DataFrame(rows)
    column_order = ["variant", "status", "best_val_dvh", "final_val_dvh",
                    "MAE", "DVH", "D95_PTV70", "D95_PTV63", "D95_PTV56",
                    "Brainstem(54Gy)", "SpinalCord(45Gy)", "Mandible(70Gy)",
                    "LeftParotid(26Gy)", "RightParotid(26Gy)"]
    column_order = [c for c in column_order if c in summary_df.columns]
    summary_df = summary_df[column_order]
    summary_csv = sweep_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n[sweep_reward_shaping] summary "
          "(clinical metrics from evaluate.py against each variant's best.pt)")
    print(summary_df.to_string(index=False))
    print(f"\n[sweep_reward_shaping] saved {summary_csv}")
    print(f"[sweep_reward_shaping] saved {comparison_png}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                            choices=list(VARIANTS),
                            help="which variants to run, in order")
    arg_parser.add_argument("--episodes", type=int, default=2000,
                            help="episodes per variant -- early stopping "
                                 "(cfg.early_stop_patience_evals) will cut "
                                 "this short if a variant plateaus")
    arg_parser.add_argument("--device", default="mps",
                            help="device for every variant (cpu/mps/cuda)")
    arg_parser.add_argument("--eval-split", default=None,
                            help="passed to evaluate.py --split (defaults "
                                 "to cfg.eval_split)")
    arg_parser.add_argument("--sweep-root", default="runs/sweep_reward_shaping")
    arg_parser.add_argument("--warmstart-ckpt", default="runs/warmstart.pt")
    arg_parser.add_argument("--force-warmstart", action="store_true",
                            help="recompute the shared warm-start checkpoint "
                                 "even if it already exists")
    args = arg_parser.parse_args()

    config_path = Path(args.config)
    base_config = yaml.safe_load(config_path.read_text())
    warmstart_ckpt = Path(args.warmstart_ckpt)
    sweep_root = Path(args.sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    _ensure_warmstart(config_path, warmstart_ckpt, force=args.force_warmstart)

    variant_dirs: dict[str, Path] = {}
    for variant_name in args.variants:
        variant_dirs[variant_name] = _run_candidate(
            base_config, variant_name, VARIANTS[variant_name],
            args.episodes, warmstart_ckpt, sweep_root, args.device,
        )

    _summarize(variant_dirs, sweep_root, args.eval_split)


if __name__ == "__main__":
    main()
