"""Exact state transition evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from wdlm.data.toy_world import canonicalize_state
from wdlm.schemas import StatePredictionRecord, WorldState
from wdlm.utils.io import read_jsonl


class ExactStateMetrics(BaseModel):
    """Aggregate exact-match metrics for predicted states."""

    model_config = ConfigDict(extra="forbid")

    example_count: int
    exact_match_count: int
    exact_match_accuracy: float
    field_match_count: int
    field_count: int
    per_field_accuracy: float


def flatten_state(state: WorldState | Mapping[str, Any]) -> dict[str, str]:
    """Flatten a state into deterministic leaf fields."""

    current_state = canonicalize_state(state)
    flat: dict[str, str] = {}
    for index, location_name in enumerate(current_state.locations):
        flat[f"locations.{index}"] = location_name
    for index, owner_name in enumerate(current_state.owners):
        flat[f"owners.{index}"] = owner_name
    for container_name, status in current_state.containers.items():
        flat[f"containers.{container_name}"] = status
    for object_name, object_state in current_state.objects.items():
        flat[f"objects.{object_name}.holder"] = object_state.holder
        flat[f"objects.{object_name}.visibility"] = object_state.visibility
    return flat


def _load_state_map(path: Path) -> dict[str, WorldState]:
    state_map: dict[str, WorldState] = {}
    for row in read_jsonl(path):
        record = StatePredictionRecord.model_validate(
            {"example_id": row["example_id"], "state_after": row["state_after"]}
        )
        if record.example_id in state_map:
            raise ValueError(f"Duplicate example_id found in {path}: {record.example_id}")
        state_map[record.example_id] = canonicalize_state(record.state_after)
    return state_map


def evaluate_exact_state_maps(
    *,
    gold_states: dict[str, WorldState],
    pred_states: dict[str, WorldState],
) -> ExactStateMetrics:
    """Compare predicted states with gold states."""

    gold_ids = set(gold_states)
    pred_ids = set(pred_states)
    if gold_ids != pred_ids:
        missing = sorted(gold_ids - pred_ids)
        extra = sorted(pred_ids - gold_ids)
        raise ValueError(
            f"Prediction/example_id mismatch. Missing={missing[:5]} Extra={extra[:5]}"
        )

    exact_match_count = 0
    field_match_count = 0
    field_count = 0

    for example_id in sorted(gold_states):
        gold_flat = flatten_state(gold_states[example_id])
        pred_flat = flatten_state(pred_states[example_id])
        if gold_flat == pred_flat:
            exact_match_count += 1
        field_count += len(gold_flat)
        for field_name, gold_value in gold_flat.items():
            if pred_flat.get(field_name) == gold_value:
                field_match_count += 1

    example_count = len(gold_states)
    return ExactStateMetrics(
        example_count=example_count,
        exact_match_count=exact_match_count,
        exact_match_accuracy=exact_match_count / example_count if example_count else 0.0,
        field_match_count=field_match_count,
        field_count=field_count,
        per_field_accuracy=field_match_count / field_count if field_count else 0.0,
    )


def evaluate_exact_state_files(*, gold_path: Path, pred_path: Path) -> ExactStateMetrics:
    """Load gold and predicted JSONL files, then compute exact-state metrics."""

    gold_states = _load_state_map(gold_path)
    pred_states = _load_state_map(pred_path)
    return evaluate_exact_state_maps(gold_states=gold_states, pred_states=pred_states)
