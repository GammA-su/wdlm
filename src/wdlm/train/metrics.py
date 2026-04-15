"""Metric helpers for training and benchmark evaluation."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from wdlm.train.dataset import StateTensorizer


def tensor_metrics_to_floats(metrics: Mapping[str, Tensor | float]) -> dict[str, float]:
    """Convert scalar metric values into plain floats."""

    output: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            output[key] = float(value.detach().cpu().item())
        else:
            output[key] = float(value)
    return output


def mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of flat metric rows."""

    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def exact_state_accuracy(
    predicted_state: Tensor,
    target_state: Tensor,
    *,
    state_tensorizer: StateTensorizer,
) -> float:
    """Compute exact next-state accuracy from explicit state predictions."""

    decoded_prediction = state_tensorizer.decode_scores(predicted_state)
    matches = (decoded_prediction == target_state).all(dim=1)
    return float(matches.float().mean().item()) if matches.numel() else 0.0


def per_field_accuracy(
    predicted_state: Tensor,
    target_state: Tensor,
    *,
    state_tensorizer: StateTensorizer,
) -> dict[str, float]:
    """Compute holder, visibility, container, and macro accuracies."""

    decoded_prediction = state_tensorizer.decode_scores(predicted_state)
    grouped: dict[str, list[float]] = {
        "holder_accuracy": [],
        "visibility_accuracy": [],
        "container_accuracy": [],
    }
    for _, field_kind, field_slice in state_tensorizer.iter_field_slices():
        field_match = (
            decoded_prediction[:, field_slice] == target_state[:, field_slice]
        ).all(dim=1).float().mean().item()
        if field_kind == "holder":
            grouped["holder_accuracy"].append(float(field_match))
        elif field_kind == "visibility":
            grouped["visibility_accuracy"].append(float(field_match))
        else:
            grouped["container_accuracy"].append(float(field_match))
    output = {
        metric_name: mean(values) if values else 0.0
        for metric_name, values in grouped.items()
    }
    output["macro_accuracy"] = mean(output.values()) if output else 0.0
    return output


def paraphrase_delta_cosine(
    delta_features: Tensor,
    paraphrase_delta_features: Tensor,
    paraphrase_owner_indices: Tensor,
) -> float:
    """Average cosine similarity between each anchor and its paraphrase updates."""

    if paraphrase_delta_features.numel() == 0:
        return 0.0
    anchors = delta_features.index_select(0, paraphrase_owner_indices)
    cosine = F.cosine_similarity(anchors, paraphrase_delta_features, dim=-1)
    return float(cosine.mean().item()) if cosine.numel() else 0.0


def delta_norm_stats(delta_features: Tensor) -> dict[str, float]:
    """Return mean and population standard deviation of update norms."""

    if delta_features.numel() == 0:
        return {"delta_norm_mean": 0.0, "delta_norm_std": 0.0}
    norms = delta_features.norm(dim=-1).detach().cpu().tolist()
    if not norms:
        return {"delta_norm_mean": 0.0, "delta_norm_std": 0.0}
    return {
        "delta_norm_mean": float(mean(norms)),
        "delta_norm_std": float(pstdev(norms) if len(norms) > 1 else 0.0),
    }


def flatten_metric_dict(metrics: Mapping[str, float | Mapping[str, float]]) -> dict[str, float]:
    """Flatten nested metric dictionaries for JSONL logging."""

    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            nested = flatten_metric_dict({f"{key}.{nested_key}": nested_value for nested_key, nested_value in value.items()})
            flat.update(nested)
        else:
            if isinstance(value, Tensor):
                flat[key] = float(value.detach().cpu().item())
                continue
            flat[key] = float(value)
    return flat
