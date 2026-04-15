from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from wdlm.analysis.dataset_report import analyze_dataset_files
from wdlm.data.generate import generate_toy_world_file
from wdlm.data.ood import build_ood_splits_file
from wdlm.data.queries import generate_query_file
from wdlm.data.trajectory import generate_trajectory_files


pytestmark = pytest.mark.skipif(
    os.environ.get("WDLM_RUN_INTEGRATION_TESTS") != "1",
    reason="Reproducibility pipeline test is opt-in. Set WDLM_RUN_INTEGRATION_TESTS=1 to enable.",
)


def _build_bundle(root: Path, *, seed: int) -> dict[str, Path]:
    data_dir = root / "data"
    splits_dir = data_dir / "splits"
    outputs = {
        "single_steps": data_dir / "single_steps.jsonl",
        "steps": data_dir / "all_steps.jsonl",
        "trajectories": data_dir / "all_trajectories.jsonl",
        "qa": data_dir / "qa.jsonl",
    }
    generate_toy_world_file(out=outputs["single_steps"], num_examples=8, seed=seed)
    generate_trajectory_files(
        out_steps=outputs["steps"],
        out_trajectories=outputs["trajectories"],
        num_trajectories=6,
        seed=seed,
    )
    generate_query_file(input_steps_path=outputs["steps"], out_path=outputs["qa"], seed=seed)
    build_ood_splits_file(input_path=outputs["steps"], out_dir=splits_dir, seed=seed)
    outputs.update(
        {
            split_name: splits_dir / f"{split_name}.jsonl"
            for split_name in (
                "train",
                "val",
                "test_iid",
                "test_lexical_ood",
                "test_compositional_ood",
                "test_length_ood",
                "test_paraphrase_ood",
            )
        }
    )
    return outputs


def test_generation_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    bundle_one = _build_bundle(tmp_path / "run_one", seed=42)
    bundle_two = _build_bundle(tmp_path / "run_two", seed=42)

    for key in sorted(bundle_one):
        assert bundle_one[key].read_text(encoding="utf-8") == bundle_two[key].read_text(
            encoding="utf-8"
        ), key

    report_one = analyze_dataset_files(
        steps_path=bundle_one["steps"],
        split_dir=(tmp_path / "run_one" / "data" / "splits"),
    )
    report_two = analyze_dataset_files(
        steps_path=bundle_two["steps"],
        split_dir=(tmp_path / "run_two" / "data" / "splits"),
    )
    assert json.dumps(report_one.model_dump(mode="json"), sort_keys=True) == json.dumps(
        report_two.model_dump(mode="json"),
        sort_keys=True,
    )
