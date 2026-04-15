from __future__ import annotations

from wdlm.data.ood import build_ood_splits, composition_signature
from wdlm.data.trajectory import flatten_trajectory_steps, generate_trajectory_records


def test_ood_splits_avoid_train_leakage() -> None:
    trajectories = generate_trajectory_records(num_trajectories=9, seed=42)
    steps = flatten_trajectory_steps(trajectories)
    splits, summary = build_ood_splits(steps, seed=42)

    all_ids = [
        example.example_id
        for split_examples in splits.values()
        for example in split_examples
    ]
    assert len(all_ids) == len(set(all_ids))

    blocked_template_ids = set(summary.lexical_holdout_template_ids) | set(
        summary.paraphrase_holdout_template_ids
    )
    train_examples = splits["train"]
    assert train_examples
    assert all(example.metadata.template_id not in blocked_template_ids for example in train_examples)
    assert all(
        blocked_template_ids.isdisjoint(example.metadata.paraphrase_template_ids)
        for example in train_examples
    )
    assert all(
        composition_signature(example) not in set(summary.compositional_holdouts)
        for example in train_examples
    )
    assert all(example.episode_length <= summary.length_threshold for example in train_examples)
    assert all(
        example.episode_length > summary.length_threshold
        for example in splits["test_length_ood"]
    )
    assert any(
        example.example_id.endswith("--paraphrase-ood")
        for example in splits["test_paraphrase_ood"]
    )
