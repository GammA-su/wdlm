from __future__ import annotations

from wdlm.analysis.dataset_report import analyze_examples
from wdlm.data.ood import build_ood_splits
from wdlm.data.trajectory import flatten_trajectory_steps, generate_trajectory_records


def test_dataset_analysis_reports_expected_keys() -> None:
    trajectories = generate_trajectory_records(num_trajectories=6, seed=21)
    steps = flatten_trajectory_steps(trajectories)
    splits, _ = build_ood_splits(steps, seed=21)
    report = analyze_examples(steps, split_examples=splits)

    assert report.example_count == len(steps)
    assert report.trajectory_count == 6
    assert report.counts_by_action_type
    assert report.counts_by_template
    assert report.counts_by_difficulty
    assert report.average_paraphrases_per_example > 0
    assert report.negative_type_distribution
    assert "test_lexical_ood" in report.ood_split_statistics
