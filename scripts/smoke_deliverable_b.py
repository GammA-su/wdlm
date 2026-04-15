"""End-to-end smoke test for WDLM Deliverable B."""

from __future__ import annotations

import tempfile
from pathlib import Path

from wdlm.analysis.dataset_report import analyze_dataset_files
from wdlm.analysis.validate_dataset import validate_dataset_artifacts
from wdlm.data.generate import generate_toy_world_file
from wdlm.data.ood import build_ood_splits_file
from wdlm.data.queries import generate_query_file
from wdlm.data.trajectory import generate_trajectory_files
from wdlm.eval.exact_state import evaluate_exact_state_files


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Expected output file is missing: {path}")


def main() -> None:
    """Run the full Deliverable B smoke workflow."""

    seed = 42
    with tempfile.TemporaryDirectory(prefix="wdlm-deliverable-b-") as temp_dir:
        root = Path(temp_dir)
        data_dir = root / "data" / "toy_world"
        splits_dir = data_dir / "splits"

        single_steps = data_dir / "single_steps.jsonl"
        all_steps = data_dir / "all_steps.jsonl"
        all_trajectories = data_dir / "all_trajectories.jsonl"
        qa_path = data_dir / "qa.jsonl"

        generate_toy_world_file(out=single_steps, num_examples=8, seed=seed)
        generate_trajectory_files(
            out_steps=all_steps,
            out_trajectories=all_trajectories,
            num_trajectories=6,
            seed=seed,
        )
        generate_query_file(input_steps_path=all_steps, out_path=qa_path, seed=seed)
        build_ood_splits_file(input_path=all_steps, out_dir=splits_dir, seed=seed)
        report = analyze_dataset_files(steps_path=all_steps, split_dir=splits_dir)
        validation_report = validate_dataset_artifacts(
            steps_path=all_steps,
            trajectories_path=all_trajectories,
            qa_path=qa_path,
            split_dir=splits_dir,
            seed=seed,
        )
        metrics = evaluate_exact_state_files(
            gold_path=splits_dir / "test_iid.jsonl",
            pred_path=splits_dir / "test_iid.jsonl",
        )

        expected_files = [
            single_steps,
            all_steps,
            all_trajectories,
            qa_path,
            splits_dir / "train.jsonl",
            splits_dir / "val.jsonl",
            splits_dir / "test_iid.jsonl",
            splits_dir / "test_lexical_ood.jsonl",
            splits_dir / "test_compositional_ood.jsonl",
            splits_dir / "test_length_ood.jsonl",
            splits_dir / "test_paraphrase_ood.jsonl",
        ]
        for path in expected_files:
            _assert_exists(path)

        if metrics.exact_match_accuracy != 1.0:
            raise AssertionError("Gold-vs-gold exact-state eval did not return 1.0.")

        print(
            "Deliverable B smoke passed:",
            f"steps={report.example_count}",
            f"trajectories={validation_report.trajectory_count}",
            f"qa={validation_report.qa_count}",
            f"iid_exact={metrics.exact_match_accuracy:.1f}",
        )


if __name__ == "__main__":
    main()
