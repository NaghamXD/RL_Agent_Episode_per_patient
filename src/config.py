"""Project-wide constants and config loading.

Defaults mirror configs/default.yaml; YAML overrides apply at runtime.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch


# Fixed by OpenKBP (do NOT change)
NX = 128
NY = 128

# Organ list — order defines mask-channel order in the state tensor
PTV_NAMES = ["PTV70", "PTV63", "PTV56"]
OAR_NAMES = ["Brainstem", "SpinalCord", "Mandible", "LeftParotid", "RightParotid"]
ALL_STRUCTURES = PTV_NAMES + OAR_NAMES


@dataclass
class Config:
    raw_dir: str = "data/original"
    processed_dir: str = "data/processed"
    train_split: str = "train"
    eval_split: str = "validation"
    grid: int = 64
    n_fractions: int = 35

    n_beams: int = 9
    beamlet_h: int = 16
    beamlet_w: int = 16

    # Include the precomputed per-patient beam-path channel (1 of 12) in the
    # state tensor. False drops it -> in_channels becomes 11. in_channels is
    # inferred at runtime from _build_state's actual output shape (see
    # train.py/evaluate.py: `in_channels = initial_state.shape[0]`), so this
    # flag alone is sufficient to resize the network -- no other code change
    # needed. Checkpoints are NOT compatible across different values of this
    # flag (different in_channels -> different first Conv3d shape).
    include_beam_paths: bool = True

    prescription: Dict[str, float] = field(default_factory=lambda: {
        "PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0,
    })
    oar_tolerance: Dict[str, float] = field(default_factory=lambda: {
        "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
        "LeftParotid": 26.0, "RightParotid": 26.0,
    })
    # Per-OAR multiplier inside the OAR penalty (1.0 = neutral).
    oar_weights: Dict[str, float] = field(default_factory=lambda: {
        "Brainstem": 1.0, "SpinalCord": 1.0, "Mandible": 1.0,
        "LeftParotid": 1.0, "RightParotid": 1.0,
    })
    # Serial OARs get an additional Dmax term (single hot voxel is catastrophic).
    oar_serial: Dict[str, bool] = field(default_factory=lambda: {
        "Brainstem": True, "SpinalCord": True, "Mandible": False,
        "LeftParotid": False, "RightParotid": False,
    })

    lambda_oar: float = 1.0
    # Sub-weights inside the OAR penalty: per-voxel is the main signal,
    # mean is a small stabiliser, dmax fires only for serial OARs.
    oar_voxel_subweight: float = 1.0
    oar_mean_subweight:  float = 0.2
    oar_dmax_subweight:  float = 1.0

    # Per-fraction PTV reward weight.  Each fraction is its own PPO
    # episode and the agent is rewarded for delivering a fair share of
    # the remaining PTV gap (see reward.fractional_ptv_reward).
    lambda_ptv: float = 1.0

    # --- sequential (multi-step MDP) mode -------------------------------
    # When True, one *episode == one patient course* (35 fractions): the
    # env returns ``done`` only on the final fraction, the reward becomes
    # dense potential-based shaping + a terminal whole-course objective,
    # and PPO runs real GAE so ``gamma`` / ``gae_lambda`` are LIVE and do
    # temporal credit assignment. When False the legacy contextual-bandit
    # path is used (each fraction is its own 1-step episode; gamma inert).
    sequential: bool = True
    # Physical per-fraction dose cap (Gy/voxel). This is what forces the
    # agent to *spread* dose across the 35 fractions and makes the
    # allocation genuinely sequential rather than a one-shot dump. 0 = off.
    max_fraction_dose: float = 3.0
    # Scale of the PTV-progress potential used for dense shaping. The
    # shaping term is gamma*phi(s') - phi(s) and is policy-invariant
    # (Ng et al. 1999), so it speeds learning without biasing the optimum.
    lambda_phi: float = 1.0
    # Weight on the (negative) DVH score inside the terminal reward. The
    # terminal reward = lambda_ptv*coverage - terminal_dvh_weight*dvh_score.
    terminal_dvh_weight: float = 0.1
    # --- reward-shaping ablation (see src/env/reward.py) ---
    # All three default to the old (pre-improvement) behaviour; see
    # scripts/sweep_reward_shaping.py for the actual A/B/C comparison.
    # Improvement A: ptv_gap_fraction's power (1.0=old linear, 2.0=quadratic).
    ptv_gap_power: float = 1.0
    # Improvement B: oar_overshoot_fraction's soft-barrier steepness
    # (None=old hard threshold; a float enables the soft barrier).
    oar_barrier_steepness: Optional[float] = None
    # Improvement C: use soft_coverage (differentiable) instead of the hard
    # coverage() step inside terminal_reward.
    terminal_use_soft_coverage: bool = False
    # Number of patient trajectories collected per PPO update in sequential
    # mode (advantages are normalised across the whole batch).
    batch_n_patients: int = 8

    # --- actor exploration (log-std) ---
    # Initial value of the global, state-independent log-std. The actor's mu
    # is warm-started onto the static FISTA-NNLS plan, so this MUST start
    # reasonably high (sigma = exp(init)) or PPO stays pinned to that
    # one-shot baseline and never explores multi-step trade-offs. 0.0 => sigma
    # = 1.0. ``actor_log_std_max`` is the clamp ceiling applied every forward.
    actor_log_std_init: float = 0.0
    actor_log_std_max:  float = 2.0

    # gamma / gae_lambda are LIVE in sequential mode (real GAE over the
    # 35-fraction trajectory). They remain inert in the legacy bandit path
    # (DoseEnv.step returns done=True every fraction -> GAE collapses to
    # advantage = reward - value). See PPO._compute_gae.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    ppo_epochs: int = 4
    minibatch: int = 64

    total_episodes: int = 2000
    eval_every: int = 50
    seed: int = 42
    # "auto" picks the best available accelerator (cuda > mps > cpu).
    # Set explicitly ("cuda" / "mps" / "cpu") to force one; an unavailable
    # explicit choice falls back to cpu with a printed warning. This
    # dataclass default is just a fallback for a YAML that omits the key --
    # configs/default.yaml pins "cpu" explicitly with the reasoning
    # (a reproducible MPS NaN bug on this architecture); read that comment
    # before relying on "auto"/"mps" on Apple Silicon.
    device: str = "auto"
    ckpt_dir: str = "runs"

    # --- warm-start (supervised actor pretraining) ---
    # If True, before PPO begins we load the cached per-patient warm-start
    # action a* (from scripts/compute_warmstart_actions.py) and fit the
    # actor head with MSE on inv_softplus(a*) for ``warmstart_epochs``
    # epochs. This breaks the symmetry of the uniform-init policy so PPO
    # has a per-beamlet starting plan to refine instead of a constant
    # ``softplus(-1) ~ 0.31`` across every beamlet.
    warmstart_enabled: bool = True
    warmstart_epochs: int = 10
    warmstart_minibatch: int = 4
    warmstart_lr: float = 3e-4

    # --- OAR-weight curriculum ---
    # During the first ``lambda_oar_ramp_episodes`` patients of PPO
    # training, the *effective* ``lambda_oar`` used by the env reward
    # ramps linearly from
    #     lambda_oar * lambda_oar_ramp_start_factor
    # up to the full ``lambda_oar``. This lets the agent learn to
    # actually deliver dose first (PTV reward dominates) before being
    # heavily penalised for incidental OAR irradiation; without it PPO
    # tends to converge to a near-zero-action policy because the
    # marginal OAR penalty exceeds the marginal PTV reward at
    # initialisation. Set ``lambda_oar_ramp_episodes: 0`` to disable.
    lambda_oar_ramp_episodes: int = 200
    lambda_oar_ramp_start_factor: float = 0.25

    # --- best.pt selection ---
    # ``best.pt`` is selected by *validation DVH score* (lower is better)
    # evaluated deterministically every ``best_eval_every`` patients on
    # the first ``best_n_val_patients`` patients of ``cfg.eval_split``.
    # The legacy training-reward rolling-mean criterion was prone to
    # picking a near-zero-dose early policy whose reward was less
    # negative simply because OAR penalties were small.
    best_n_val_patients: int = 3
    best_eval_every: int = 25
    # Kept for back-compat / diagnostics; no longer drives best.pt.
    best_rolling_window: int = 10

    # --- plateau monitor (diagnostic; never auto-stops training) ---
    # Number of consecutive validation checks (each spaced best_eval_every
    # episodes apart) with no >min_delta relative improvement in val_dvh
    # before a "[plateau]" message is logged. 0 disables the monitor
    # entirely (default off).
    early_stop_patience_evals: int = 0
    early_stop_min_delta: float = 0.005

    @property
    def n_beamlets(self) -> int:
        return self.n_beams * self.beamlet_h * self.beamlet_w

    @property
    def state_channels(self) -> int:
        # 1 CT + len(structures) masks + cumulative_dose + ptv_dose_gap + beam_paths
        return 1 + len(ALL_STRUCTURES) + 1 + 1 + 1   # = 11 here -> see note

    @property
    def n_voxels(self) -> int:
        return self.grid ** 3


def resolve_device(requested: str = "auto") -> torch.device:
    """Map ``cfg.device`` to an actually-available ``torch.device``.

    ``"auto"`` picks the best available accelerator: CUDA, then Apple
    Metal (MPS), then CPU. An explicit ``"cuda"`` / ``"mps"`` request
    that isn't available on this machine falls back to CPU (with a
    warning) rather than crashing.
    """
    requested = (requested or "auto").strip().lower()
    mps_available = torch.backends.mps.is_available()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device] cuda requested but not available; falling back to cpu")
        return torch.device("cpu")

    if requested == "mps":
        if mps_available:
            return torch.device("mps")
        print("[device] mps requested but not available; falling back to cpu")
        return torch.device("cpu")

    return torch.device("cpu")


def load_config(path: str | Path) -> Config:
    with open(path, "r") as yaml_file:
        cfg_dict = yaml.safe_load(yaml_file)
    # Back-compat: old configs used `lambda_shaping` and `dvh_bonus_scale`.
    # The new per-fraction design replaces them with `lambda_ptv` and drops
    # the terminal DVH bonus entirely.  Silently migrate so existing YAMLs
    # keep loading.
    if "lambda_shaping" in cfg_dict and "lambda_ptv" not in cfg_dict:
        cfg_dict["lambda_ptv"] = cfg_dict.pop("lambda_shaping")
    cfg_dict.pop("lambda_shaping", None)
    cfg_dict.pop("dvh_bonus_scale", None)
    # Back-compat: short sub-weight names (``oar_voxel_w`` / ``oar_mean_w`` /
    # ``oar_dmax_w``) were renamed to descriptive ones for readability.
    legacy_subweight_keys = (
        ("oar_voxel_w", "oar_voxel_subweight"),
        ("oar_mean_w",  "oar_mean_subweight"),
        ("oar_dmax_w",  "oar_dmax_subweight"),
    )
    for legacy_key, descriptive_key in legacy_subweight_keys:
        if legacy_key in cfg_dict and descriptive_key not in cfg_dict:
            cfg_dict[descriptive_key] = cfg_dict.pop(legacy_key)
        cfg_dict.pop(legacy_key, None)
    return Config(**cfg_dict)
