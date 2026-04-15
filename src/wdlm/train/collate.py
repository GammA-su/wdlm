"""Batch collation helpers."""

from __future__ import annotations

import torch


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row_index, sequence in enumerate(sequences):
        seq_len = len(sequence)
        input_ids[row_index, :seq_len] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, :seq_len] = True
    return input_ids, attention_mask


def collate_examples(
    samples: list[dict[str, object]],
    *,
    pad_id: int,
    max_paraphrases: int,
) -> dict[str, torch.Tensor | list[str]]:
    """Collate dataset items into a training batch."""

    example_ids = [str(sample["example_id"]) for sample in samples]
    input_ids, attention_mask = _pad_sequences(
        [list(sample["input_ids"]) for sample in samples],
        pad_id,
    )
    labels = input_ids.clone()
    state_before = torch.stack([sample["state_before"] for sample in samples])  # type: ignore[arg-type]
    state_after = torch.stack([sample["state_after"] for sample in samples])  # type: ignore[arg-type]
    action_type_ids = torch.tensor(
        [int(sample["action_type_id"]) for sample in samples],
        dtype=torch.long,
    )

    paraphrase_sequences: list[list[int]] = []
    paraphrase_owner_indices: list[int] = []
    for sample_index, sample in enumerate(samples):
        paraphrases = list(sample["paraphrase_ids"])[:max_paraphrases]
        for paraphrase in paraphrases:
            paraphrase_sequences.append(list(paraphrase))
            paraphrase_owner_indices.append(sample_index)
    if paraphrase_sequences:
        paraphrase_input_ids, paraphrase_attention_mask = _pad_sequences(paraphrase_sequences, pad_id)
        paraphrase_owner_indices_tensor = torch.tensor(paraphrase_owner_indices, dtype=torch.long)
    else:
        paraphrase_input_ids = torch.full((0, 1), pad_id, dtype=torch.long)
        paraphrase_attention_mask = torch.zeros((0, 1), dtype=torch.bool)
        paraphrase_owner_indices_tensor = torch.zeros((0,), dtype=torch.long)

    negative_sequences: list[list[int]] = []
    negative_owner_indices: list[int] = []
    for sample_index, sample in enumerate(samples):
        negatives = list(sample["negative_ids"])[:max_paraphrases]
        for negative in negatives:
            negative_sequences.append(list(negative))
            negative_owner_indices.append(sample_index)
    if negative_sequences:
        negative_input_ids, negative_attention_mask = _pad_sequences(negative_sequences, pad_id)
        negative_owner_indices_tensor = torch.tensor(negative_owner_indices, dtype=torch.long)
    else:
        negative_input_ids = torch.full((0, 1), pad_id, dtype=torch.long)
        negative_attention_mask = torch.zeros((0, 1), dtype=torch.bool)
        negative_owner_indices_tensor = torch.zeros((0,), dtype=torch.long)

    return {
        "example_ids": example_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "state_before": state_before,
        "state_after": state_after,
        "action_type_ids": action_type_ids,
        "paraphrase_input_ids": paraphrase_input_ids,
        "paraphrase_attention_mask": paraphrase_attention_mask,
        "paraphrase_owner_indices": paraphrase_owner_indices_tensor,
        "negative_input_ids": negative_input_ids,
        "negative_attention_mask": negative_attention_mask,
        "negative_owner_indices": negative_owner_indices_tensor,
    }
