from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wdlm.cli import app
from wdlm.data.ood import build_ood_splits_file
from wdlm.data.queries import generate_query_file
from wdlm.data.trajectory import generate_trajectory_files
from wdlm.utils.io import read_jsonl, write_jsonl


pytestmark = pytest.mark.skipif(
    os.environ.get("WDLM_RUN_INTEGRATION_TESTS") != "1",
    reason="Dataset validation integration test is opt-in. Set WDLM_RUN_INTEGRATION_TESTS=1 to enable.",
)


runner = CliRunner()


def _last_json_line(output: str) -> dict[str, object]:
    return json.loads(output.strip().splitlines()[-1])


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    steps_path = tmp_path / "all_steps.jsonl"
    trajectories_path = tmp_path / "all_trajectories.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    split_dir = tmp_path / "splits"
    generate_trajectory_files(
        out_steps=steps_path,
        out_trajectories=trajectories_path,
        num_trajectories=4,
        seed=42,
    )
    generate_query_file(input_steps_path=steps_path, out_path=qa_path, seed=42)
    build_ood_splits_file(input_path=steps_path, out_dir=split_dir, seed=42)
    return steps_path, trajectories_path, qa_path, split_dir


def test_validate_dataset_exits_successfully_on_clean_data(tmp_path: Path) -> None:
    steps_path, trajectories_path, qa_path, split_dir = _build_artifacts(tmp_path)

    result = runner.invoke(
        app,
        [
            "validate-dataset",
            "--steps",
            str(steps_path),
            "--trajectories",
            str(trajectories_path),
            "--qa",
            str(qa_path),
            "--split-dir",
            str(split_dir),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _last_json_line(result.stdout)
    assert payload["step_count"] > 0
    assert payload["trajectory_count"] > 0
    assert payload["qa_count"] > 0


def test_validate_dataset_catches_corrupted_record(tmp_path: Path) -> None:
    steps_path, trajectories_path, qa_path, split_dir = _build_artifacts(tmp_path)
    corrupt_steps_path = tmp_path / "corrupt_steps.jsonl"
    rows = read_jsonl(steps_path)
    rows[0]["state_after"] = rows[0]["state_before"]
    write_jsonl(corrupt_steps_path, rows)

    result = runner.invoke(
        app,
        [
            "validate-dataset",
            "--steps",
            str(corrupt_steps_path),
            "--trajectories",
            str(trajectories_path),
            "--qa",
            str(qa_path),
            "--split-dir",
            str(split_dir),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 1
    assert "state_after does not match applied action" in result.output
