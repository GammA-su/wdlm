from __future__ import annotations

from pathlib import Path

import pytest

from wdlm.eval.exact_state import evaluate_exact_state_files
from wdlm.utils.io import write_jsonl


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    write_jsonl(path, rows)


def _state_with(holder: str) -> dict[str, object]:
    return {
        "locations": ["floor", "shelf", "table"],
        "owners": ["alice", "bob"],
        "containers": {"cabinet": "closed", "drawer": "open"},
        "objects": {
            "apple": {"holder": "table", "visibility": "visible"},
            "blue_coin": {"holder": holder, "visibility": "visible"},
            "book": {"holder": "alice", "visibility": "hidden"},
            "red_key": {"holder": "shelf", "visibility": "visible"},
        },
    }


def test_exact_state_evaluator_reports_full_match(tmp_path: Path) -> None:
    rows = [{"example_id": "example-1", "state_after": _state_with("drawer")}]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_dataset(gold_path, rows)
    _write_dataset(pred_path, rows)

    metrics = evaluate_exact_state_files(gold_path=gold_path, pred_path=pred_path)
    assert metrics.exact_match_accuracy == 1.0
    assert metrics.per_field_accuracy == 1.0


def test_exact_state_evaluator_reports_partial_mismatch(tmp_path: Path) -> None:
    gold_rows = [{"example_id": "example-1", "state_after": _state_with("drawer")}]
    pred_rows = [{"example_id": "example-1", "state_after": _state_with("table")}]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_dataset(gold_path, gold_rows)
    _write_dataset(pred_path, pred_rows)

    metrics = evaluate_exact_state_files(gold_path=gold_path, pred_path=pred_path)
    assert metrics.exact_match_accuracy == 0.0
    assert metrics.per_field_accuracy == pytest.approx(14 / 15)
