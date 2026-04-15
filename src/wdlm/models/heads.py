"""Small neural heads used by baseline and WDLM models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanPooler(nn.Module):
    """Pool token hidden states into one chunk representation."""

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom


class DeltaHead(nn.Module):
    """Predict a latent delta from the previous state and chunk representation."""

    def __init__(self, *, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim + state_dim),
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, chunk_repr: Tensor, state_repr: Tensor) -> Tensor:
        return self.net(torch.cat([chunk_repr, state_repr], dim=-1))


class TransitionHead(nn.Module):
    """Predict a latent next-state representation without explicit delta modeling."""

    def __init__(self, *, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim + state_dim),
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, chunk_repr: Tensor, state_repr: Tensor) -> Tensor:
        return self.net(torch.cat([chunk_repr, state_repr], dim=-1))


class StateEncoder(nn.Module):
    """Project compact explicit state vectors into latent state space."""

    def __init__(self, *, input_dim: int, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state_tensor: Tensor) -> Tensor:
        return self.net(state_tensor)


class StateDecoder(nn.Module):
    """Decode a latent state representation into explicit state logits."""

    def __init__(self, *, state_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent_state: Tensor) -> Tensor:
        return self.net(latent_state)


class ProjectionHead(nn.Module):
    """Project pooled chunk features into a comparison space."""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class LMHead(nn.Module):
    """Project hidden states to vocabulary logits."""

    def __init__(self, *, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)
