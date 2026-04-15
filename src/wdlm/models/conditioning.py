"""State-conditioning layers."""

from __future__ import annotations

from torch import Tensor, nn


class AdditiveStateConditioner(nn.Module):
    """Add a projected latent state to token hidden states."""

    def __init__(self, *, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_dim)

    def forward(self, hidden_states: Tensor, latent_state: Tensor) -> Tensor:
        state_bias = self.proj(latent_state).unsqueeze(1)
        return hidden_states + state_bias
