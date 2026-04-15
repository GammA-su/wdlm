"""Loss functions for baseline and WDLM training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_text_cross_entropy(logits: Tensor, labels: Tensor, *, pad_id: int) -> Tensor:
    """Compute next-token cross-entropy with padding ignored."""

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=pad_id,
    )


def state_mse_loss(s_pred: Tensor, s_target: Tensor) -> Tensor:
    """Mean-squared latent state reconstruction loss."""

    return F.mse_loss(s_pred, s_target)


def paraphrase_delta_invariance_loss(
    delta_hat: Tensor,
    paraphrase_delta_hat: Tensor,
    paraphrase_owner_indices: Tensor,
) -> Tensor:
    """Match deltas from paraphrases to their owning positive example."""

    if paraphrase_delta_hat.numel() == 0:
        return delta_hat.new_zeros(())
    targets = delta_hat.index_select(0, paraphrase_owner_indices)
    return F.mse_loss(paraphrase_delta_hat, targets)


def supervised_contrastive_delta_loss(
    delta_hat: Tensor,
    paraphrase_delta_hat: Tensor,
    paraphrase_owner_indices: Tensor,
    negative_delta_hat: Tensor,
    negative_owner_indices: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Contrastive loss with explicit positive paraphrases and negative updates."""

    if paraphrase_delta_hat.numel() == 0 or negative_delta_hat.numel() == 0:
        return delta_hat.new_zeros(())

    anchor = F.normalize(delta_hat, dim=-1)
    positives = F.normalize(paraphrase_delta_hat, dim=-1)
    negatives = F.normalize(negative_delta_hat, dim=-1)

    losses: list[Tensor] = []
    for anchor_index in range(anchor.size(0)):
        positive_mask = paraphrase_owner_indices == anchor_index
        negative_mask = negative_owner_indices == anchor_index
        if not positive_mask.any() or not negative_mask.any():
            continue
        anchor_vector = anchor[anchor_index].unsqueeze(0)
        positive_logits = (anchor_vector @ positives[positive_mask].transpose(0, 1)).squeeze(0) / temperature
        negative_logits = (anchor_vector @ negatives[negative_mask].transpose(0, 1)).squeeze(0) / temperature
        denominator = torch.logsumexp(torch.cat([positive_logits, negative_logits], dim=0), dim=0)
        losses.append(-(positive_logits.mean() - denominator))
    if not losses:
        return delta_hat.new_zeros(())
    return torch.stack(losses).mean()
