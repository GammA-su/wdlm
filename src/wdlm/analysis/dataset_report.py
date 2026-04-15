"""Dataset analysis helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wdlm.data.split import load_examples
from wdlm.schemas import ExampleRecord


EXPECTED_SPLIT_FILES: tuple[str, ...] = (
    "train",
    "val",
    "test_iid",
    "test_lexical_ood",
    "test_compositional_ood",
    "test_length_ood",
    "test_paraphrase_ood",
)


class DatasetReport(BaseModel):
    """Aggregate summary statistics for a WDLM dataset."""

    model_config = ConfigDict(extra="forbid")

    example_count: int
    trajectory_count: int
    counts_by_action_type: dict[str, int]
    counts_by_template: dict[str, int]
    counts_by_difficulty: dict[str, int]
    average_paraphrases_per_example: float
    negative_type_distribution: dict[str, int]
    ood_split_statistics: dict[str, dict[str, int]]


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def analyze_examples(
    steps: list[ExampleRecord],
    *,
    split_examples: dict[str, list[ExampleRecord]] | None = None,
) -> DatasetReport:
    """Analyze a collection of per-step examples."""

    action_types = [example.action_struct.type for example in steps]
    templates = [example.metadata.template_id for example in steps]
    difficulties = [example.metadata.difficulty for example in steps]
    negative_types = [
        negative.negative_type
        for example in steps
        for negative in example.negative_updates
    ]
    ood_statistics: dict[str, dict[str, int]] = {}
    if split_examples is not None:
        for split_name, split_values in sorted(split_examples.items()):
            ood_statistics[split_name] = {
                "example_count": len(split_values),
                "trajectory_count": len({example.trajectory_id for example in split_values}),
            }

    average_paraphrases = (
        sum(len(example.paraphrases) for example in steps) / len(steps) if steps else 0.0
    )
    return DatasetReport(
        example_count=len(steps),
        trajectory_count=len({example.trajectory_id for example in steps}),
        counts_by_action_type=_count(action_types),
        counts_by_template=_count(templates),
        counts_by_difficulty=_count(difficulties),
        average_paraphrases_per_example=average_paraphrases,
        negative_type_distribution=_count(negative_types),
        ood_split_statistics=ood_statistics,
    )


def analyze_dataset_files(
    *,
    steps_path: Path,
    split_dir: Path | None = None,
) -> DatasetReport:
    """Analyze a steps JSONL file and optional OOD split directory."""

    steps = load_examples(steps_path)
    split_examples: dict[str, list[ExampleRecord]] | None = None
    if split_dir is not None:
        split_examples = {}
        for split_name in EXPECTED_SPLIT_FILES:
            path = split_dir / f"{split_name}.jsonl"
            if path.exists():
                split_examples[split_name] = load_examples(path)
    return analyze_examples(steps, split_examples=split_examples)
