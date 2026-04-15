"""Minimal WDLM scaffold."""

from __future__ import annotations

from torch import Tensor, nn

from wdlm.models.backbone import CausalTransformerBackbone
from wdlm.models.conditioning import AdditiveStateConditioner
from wdlm.models.heads import DeltaHead, LMHead, MeanPooler, StateDecoder, StateEncoder


class WDLMModel(nn.Module):
    """Minimal model scaffold for latent state transition prediction."""

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
        use_state_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.use_state_conditioning = use_state_conditioning
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
        self.delta_head = DeltaHead(hidden_dim=d_model, state_dim=state_dim)
        self.state_decoder = StateDecoder(
            state_dim=state_dim,
            hidden_dim=d_model,
            output_dim=state_input_dim,
        )
        self.conditioner = AdditiveStateConditioner(state_dim=state_dim, hidden_dim=d_model)
        self.lm_head = LMHead(hidden_dim=d_model, vocab_size=vocab_size)

    def encode_state(self, state_tensor: Tensor) -> Tensor:
        """Encode an explicit state vector into latent space."""

        return self.state_encoder(state_tensor)

    def compute_delta(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        state_before: Tensor,
    ) -> dict[str, Tensor]:
        """Compute chunk representation and latent delta."""

        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        chunk_repr = self.chunk_pooler(hidden_states, attention_mask)
        s_prev = self.encode_state(state_before)
        delta_hat = self.delta_head(chunk_repr, s_prev)
        s_pred = s_prev + delta_hat
        state_logits = self.state_decoder(s_pred)
        return {
            "hidden_states": hidden_states,
            "chunk_repr": chunk_repr,
            "s_prev": s_prev,
            "delta_hat": delta_hat,
            "delta_features": delta_hat,
            "s_pred": s_pred,
            "state_logits": state_logits,
        }

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        state_before: Tensor,
    ) -> dict[str, Tensor]:
        outputs = self.compute_delta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_before=state_before,
        )
        if self.use_state_conditioning:
            decoder_hidden = self.conditioner(outputs["hidden_states"], outputs["s_pred"])
        else:
            decoder_hidden = outputs["hidden_states"]
        outputs["logits"] = self.lm_head(decoder_hidden)
        return outputs
