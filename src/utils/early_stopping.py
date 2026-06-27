"""Validation-plateau monitor for ``train.py``.

``update()`` just detects a plateau (returns ``True`` once, on the check
where it's first flagged) -- ``train.py`` is what acts on that signal and
breaks the training loop. Disabled by default
(``early_stop_patience_evals: 0``).

PPO on this task can have a transient plateau during the OAR-weight
curriculum ramp before a later breakthrough at full ``lambda_oar``, so
patience only starts counting once the curriculum has fully ramped
(``start_episode``) -- otherwise the ramp's own transient dip would look
like a plateau and stop training before it really gets going. Concrete
motivating case: a 3500-episode MPS run peaked (best val_dvh) at episode
174, then -- with nothing to stop it -- ran all the way to 3500 while the
policy collapsed to a val_dvh roughly 5x worse (entropy/log_std ran away
unchecked). ``best.pt`` was never at risk (it's saved independently
whenever val_dvh improves), but ~3300 episodes of compute were wasted.
"""
from __future__ import annotations


class PlateauMonitor:
    def __init__(self, *, patience_evals: int, min_delta: float,
                start_episode: int):
        self.patience_evals = int(patience_evals)
        self.min_delta = float(min_delta)
        self.start_episode = int(start_episode)
        self.best_score = float("inf")
        self.evals_since_improvement = 0
        self.plateaued = False

    @property
    def enabled(self) -> bool:
        return self.patience_evals > 0

    def update(self, episode_index: int, score: float) -> bool:
        """Call on every validation check with the episode index and the
        validation DVH score (lower is better).

        Returns ``True`` exactly once, the validation check on which a new
        plateau is first flagged (so the caller logs it once, not on every
        subsequent stalled check).
        """
        if not self.enabled or episode_index < self.start_episode:
            self.best_score = min(self.best_score, score)
            return False

        relative_improvement = (
            (self.best_score - score) / max(abs(self.best_score), 1e-8)
        )
        if relative_improvement > self.min_delta:
            self.best_score = score
            self.evals_since_improvement = 0
            self.plateaued = False
            return False

        self.best_score = min(self.best_score, score)
        self.evals_since_improvement += 1
        if self.evals_since_improvement >= self.patience_evals and not self.plateaued:
            self.plateaued = True
            return True
        return False
