"""3-D CNN encoder: (C, G, G, G) -> (latent_dim,).

All ReLUs here are ``inplace=False`` deliberately, not by oversight:
``inplace=True`` reliably corrupts the backward pass on this machine's MPS
backend (PyTorch 2.12.1) -- gradients escalate batch over batch and the
actor's warm-start pretraining (real supervised regression on real patient
data) NaNs out within one epoch, every time, regardless of weight init
scheme. Verified by isolating exactly this flag on an otherwise-identical
forward/backward loop: ``inplace=False`` trains stably on MPS (matching
CPU's loss curve); ``inplace=True`` does not. Do not change back to
``inplace=True`` without re-verifying on MPS.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float) -> nn.Module:
    """Orthogonal weight init + zero bias -- the standard PPO recipe
    (Engstrom et al. 2020, "Implementation Matters in Deep RL"; same
    scheme used by Stable-Baselines3 / CleanRL). ``gain=sqrt(2)`` for
    layers feeding a ReLU, ``gain=1.0`` for an unsquashed scalar output
    head (e.g. the critic), ``gain=0.01`` for a policy mean head that
    should start near-zero.
    """
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0.0)
    return module


class CNNEncoder3D(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        relu_gain = nn.init.calculate_gain("relu")
        self.conv = nn.Sequential(
            orthogonal_init(nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1), relu_gain),  # 32
            nn.ReLU(inplace=False),
            orthogonal_init(nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), relu_gain),           # 16
            nn.ReLU(inplace=False),
            orthogonal_init(nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), relu_gain),           #  8
            nn.ReLU(inplace=False),
            orthogonal_init(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), relu_gain),          #  4
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = orthogonal_init(nn.Linear(128, latent_dim), relu_gain)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        convolution_output = self.conv(state).flatten(1)
        return self.proj(convolution_output)
