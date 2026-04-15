from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wdlm.cli import app
from wdlm.data.ood import build_ood_splits_file
from wdlm.data.trajectory import generate_trajectory_files


pytestmark = pytest.mark.skipif(
    os.environ.get("WDLM_RUN_INTEGRATION_TESTS") != "1",
    reason="CLI integration test is opt-in. Set WDLM_RUN_INTEGRATION_TESTS=1 to enable.",
)


runner = CliRunner()


def _last_json_line(output: str) -> dict[str, object]:
    return json.loads(output.strip().splitlines()[-1])


def _build_steps_and_splits(tmp_path: Path) -> tuple[Path, Path]:
    steps_path = tmp_path / "all_steps.jsonl"
    trajectories_path = tmp_path / "all_trajectories.jsonl"
    split_dir = tmp_path / "splits"
    generate_trajectory_files(
        out_steps=steps_path,
        out_trajectories=trajectories_path,
        num_trajectories=4,
        seed=42,
    )
    build_ood_splits_file(input_path=steps_path, out_dir=split_dir, seed=42)
    return steps_path, split_dir


def test_analyze_dataset_accepts_steps_and_input_alias(tmp_path: Path) -> None:
    steps_path, split_dir = _build_steps_and_splits(tmp_path)

    result_steps = runner.invoke(
        app,
        [
            "analyze-dataset",
            "--steps",
            str(steps_path),
            "--split-dir",
            str(split_dir),
        ],
    )
    result_input = runner.invoke(
        app,
        [
            "analyze-dataset",
            "--input",
            str(steps_path),
            "--split-dir",
            str(split_dir),
        ],
    )

    assert result_steps.exit_code == 0, result_steps.output
    assert result_input.exit_code == 0, result_input.output
    assert _last_json_line(result_steps.stdout) == _last_json_line(result_input.stdout)
