"""Actor + Critic heads for PPO.

Actor outputs a Normal distribution over raw beamlet logits; the action used
in the environment is ``softplus(sample) * action_scale`` so intensities are
non-negative and start tiny (a few %% of a Gy/beamlet) to avoid blowing up
the per-fraction dose at initialization.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .encoder import CNNEncoder3D, orthogonal_init


# Per-beamlet weight at init.  We want the *initial* policy to already
# deliver a non-trivial fractional dose so the agent can feel the
# coverage-vs-OAR trade-off, instead of getting stuck in a "do nothing"
# local optimum (reward ~0).  softplus(-1) ~= 0.31 / beamlet; with the
# DIM scale (~1 Gy/beamlet, ~2k active beamlets) this puts a fresh
# rollout's PTV D95 roughly in the 30-50 Gy range, far enough above the
# zero-policy that PPO's gradient has signal in both directions.
ACTOR_MU_BIAS_INIT = -1.0
# Default initial log-std. Note: in the SEQUENTIAL setup the actor's mu is
# warm-started onto the static FISTA-NNLS plan, so a *low* initial sigma would
# pin the policy to that one-shot baseline and prevent it from discovering
# multi-step strategic trade-offs. The init is therefore configurable
# (cfg.actor_log_std_init) and defaults higher there; this constant is only
# the fallback when no value is supplied.
ACTOR_LOG_STD_INIT = -1.0
ACTOR_LOG_STD_MAX = 2.0   # clamp ceiling (sigma <= e^2 ~ 7.4); was 1.0
ACTION_SCALE = 1.0  # multiplicative scale on softplus output


class ActorCritic(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_beamlets: int,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 log_std_init: float = ACTOR_LOG_STD_INIT,
                 log_std_max: float = ACTOR_LOG_STD_MAX):
        super().__init__()
        self.encoder = CNNEncoder3D(in_channels, latent_dim)
        relu_gain = nn.init.calculate_gain("relu")
        # +1 for the scalar fraction_progress (fraction_index / n_fractions)
        # appended after the CNN encoder output.
        feature_dim = latent_dim + 1
        self.actor_trunk = nn.Sequential(
            orthogonal_init(nn.Linear(feature_dim, hidden_dim), relu_gain), nn.ReLU(inplace=False),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), relu_gain), nn.ReLU(inplace=False),
        )
        self.actor_mu = nn.Linear(hidden_dim, n_beamlets)
        # Initialize actor head small so initial mu ~ ACTOR_MU_BIAS_INIT.
        nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)
        nn.init.constant_(self.actor_mu.bias, ACTOR_MU_BIAS_INIT)
        # state-independent log-std (standard PPO for continuous control),
        # initialized from cfg so the warm-started policy can start
        # exploratory and break away from the static baseline.
        self.log_std_max = float(log_std_max)
        self.log_std = nn.Parameter(
            torch.full((n_beamlets,), float(log_std_init))
        )

        # Standard PPO orthogonal-init recipe (Engstrom et al. 2020;
        # Stable-Baselines3 / CleanRL use the same scheme): gain=sqrt(2) on
        # hidden layers feeding a ReLU, gain=1.0 on the unsquashed scalar
        # value head (no shrinking needed -- unlike the policy mean head
        # above, there's no saturation risk to guard against).
        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(feature_dim, hidden_dim), relu_gain), nn.ReLU(inplace=False),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), relu_gain), nn.ReLU(inplace=False),
            orthogonal_init(nn.Linear(hidden_dim, 1), 1.0),
        )

    def features(self,
                 state: torch.Tensor,
                 fraction_progress: torch.Tensor) -> torch.Tensor:
        """CNN latent || fraction_progress — the input to actor & critic heads."""
        latent = self.encoder(state)
        return torch.cat([latent, fraction_progress.unsqueeze(-1)], dim=-1)

    def forward(self,
                state: torch.Tensor,
                fraction_progress: torch.Tensor):
        features_with_fraction_progress = self.features(state, fraction_progress)
        action_mean = self.actor_mu(
            self.actor_trunk(features_with_fraction_progress)
        )
        # Clamp log_std to a sane range so std is always finite and not too
        # tiny (which kills exploration) or unboundedly large. The ceiling is
        # configurable (cfg.actor_log_std_max) so the sequential agent can
        # keep a wide exploration band when it warrants it.
        action_log_std = self.log_std.clamp(min=-5.0, max=self.log_std_max)
        action_std = action_log_std.exp().expand_as(action_mean)
        policy_distribution = Normal(action_mean, action_std)
        state_value = self.critic(features_with_fraction_progress).squeeze(-1)
        return policy_distribution, state_value

    @staticmethod
    def to_action(raw_sample: torch.Tensor) -> torch.Tensor:
        """softplus -> non-negative beamlet intensities (scaled)."""
        return F.softplus(raw_sample) * ACTION_SCALE
