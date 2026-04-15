"""Small baseline causal language model."""

from __future__ import annotations

from torch import Tensor, nn

from wdlm.models.backbone import CausalTransformerBackbone
from wdlm.models.heads import LMHead, MeanPooler, ProjectionHead


class BaselineLanguageModel(nn.Module):
    """A minimal causal text model trained only with token loss."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = CausalTransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.chunk_pooler = MeanPooler()
        self.delta_projector = ProjectionHead(input_dim=d_model, output_dim=d_model)
        self.lm_head = LMHead(hidden_dim=d_model, vocab_size=vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        chunk_repr = self.chunk_pooler(hidden_states, attention_mask)
        delta_features = self.delta_projector(chunk_repr)
        return {
            "hidden_states": hidden_states,
            "chunk_repr": chunk_repr,
            "delta_features": delta_features,
            "logits": self.lm_head(hidden_states),
        }
