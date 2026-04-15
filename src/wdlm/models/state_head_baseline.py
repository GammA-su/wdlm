"""Causal LM baseline with an auxiliary state head and no delta update path."""

from __future__ import annotations

from torch import Tensor, nn

from wdlm.models.backbone import CausalTransformerBackbone
from wdlm.models.heads import LMHead, MeanPooler, ProjectionHead, StateDecoder, StateEncoder, TransitionHead


class StateHeadBaselineModel(nn.Module):
    """Predict text plus next-state logits from pooled chunk features."""

    def __init__(
        self,
        *,
        vocab_size: int,
        state_input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(
            input_dim=state_input_dim,
            hidden_dim=d_model,
            state_dim=state_dim,
        )
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
        self.transition_head = TransitionHead(hidden_dim=d_model, state_dim=state_dim)
        self.state_decoder = StateDecoder(
            state_dim=state_dim,
            hidden_dim=d_model,
            output_dim=state_input_dim,
        )
        self.delta_projector = ProjectionHead(input_dim=d_model, output_dim=state_dim)
        self.lm_head = LMHead(hidden_dim=d_model, vocab_size=vocab_size)

    def encode_state(self, state_tensor: Tensor) -> Tensor:
        """Encode an explicit state tensor into latent space."""

        return self.state_encoder(state_tensor)

    def compute_representations(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        state_before: Tensor,
    ) -> dict[str, Tensor]:
        """Compute chunk and next-state representations."""

        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        chunk_repr = self.chunk_pooler(hidden_states, attention_mask)
        s_prev = self.encode_state(state_before)
        s_pred = self.transition_head(chunk_repr, s_prev)
        state_logits = self.state_decoder(s_pred)
        delta_features = self.delta_projector(chunk_repr)
        return {
            "hidden_states": hidden_states,
            "chunk_repr": chunk_repr,
            "s_prev": s_prev,
            "s_pred": s_pred,
            "state_logits": state_logits,
            "delta_features": delta_features,
        }

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        state_before: Tensor,
    ) -> dict[str, Tensor]:
        outputs = self.compute_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_before=state_before,
        )
        outputs["logits"] = self.lm_head(outputs["hidden_states"])
        return outputs
