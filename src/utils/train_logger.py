"""CSV training-curve logger for ``train.py``.

``runs/`` is gitignored and nothing else persists a learning curve, so
neither a gamma sweep nor a plateau check has anything to compare without
this. Appends one row per episode to ``<ckpt_dir>/train_log.csv``;
opens in append mode so a ``--resume``'d run continues the same file
instead of overwriting it.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional


FIELDS = ["episode", "patient_reward", "rolling_mean_reward",
         "policy_loss", "value_loss", "entropy",
         "lambda_oar_effective", "val_dvh", "is_new_best"]


class TrainingLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self.path.exists()) or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log_episode(self, episode: int, *, patient_reward: float,
                    rolling_mean_reward: float, policy_loss: float,
                    value_loss: float, entropy: float,
                    lambda_oar_effective: float,
                    val_dvh: Optional[float] = None,
                    is_new_best: bool = False) -> None:
        self._writer.writerow({
            "episode": episode,
            "patient_reward": patient_reward,
            "rolling_mean_reward": rolling_mean_reward,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "lambda_oar_effective": lambda_oar_effective,
            "val_dvh": "" if val_dvh is None else val_dvh,
            "is_new_best": int(is_new_best),
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()
