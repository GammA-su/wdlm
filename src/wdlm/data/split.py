"""Dataset split helpers."""

from __future__ import annotations

from pathlib import Path

from wdlm.schemas import ExampleMetadata, ExampleRecord
from wdlm.utils.io import read_jsonl, write_jsonl
from wdlm.utils.rng import SeededRNG


def load_examples(path: Path) -> list[ExampleRecord]:
    """Load example records from JSONL."""

    return [ExampleRecord.model_validate(row) for row in read_jsonl(path)]


def assign_split(example: ExampleRecord, split_name: str) -> ExampleRecord:
    """Return a copy of an example with the split field updated."""

    return example.model_copy(
        update={
            "metadata": ExampleMetadata(
                split=split_name,
                template_id=example.metadata.template_id,
                paraphrase_template_ids=list(example.metadata.paraphrase_template_ids),
                difficulty=example.metadata.difficulty,
                seed=example.metadata.seed,
            )
        }
    )


def _split_trajectory_ids(
    trajectory_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, set[str]]:
    rng = SeededRNG(seed)
    shuffled_ids = rng.shuffle(trajectory_ids)
    total = len(shuffled_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": set(shuffled_ids[:train_end]),
        "val": set(shuffled_ids[train_end:val_end]),
        "test": set(shuffled_ids[val_end:]),
    }


def split_examples(
    examples: list[ExampleRecord],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[ExampleRecord]]:
    """Split examples into deterministic train/val/test partitions by trajectory."""

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    trajectory_ids = sorted({example.trajectory_id for example in examples})
    split_ids = _split_trajectory_ids(
        trajectory_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    assigned: dict[str, list[ExampleRecord]] = {"train": [], "val": [], "test": []}
    for example in examples:
        for split_name, allowed_ids in split_ids.items():
            if example.trajectory_id in allowed_ids:
                assigned[split_name].append(assign_split(example, split_name))
                break

    for split_name in assigned:
        assigned[split_name].sort(key=lambda example: example.example_id)
    return assigned


def split_dataset_file(
    *,
    input_path: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Path]:
    """Read a JSONL file, split it, and write train/val/test JSONL outputs."""

    examples = load_examples(input_path)
    splits = split_examples(
        examples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_examples_list in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        write_jsonl(
            path,
            [
                example.model_dump(mode="json", by_alias=True)
                for example in split_examples_list
            ],
        )
        paths[split_name] = path
    return paths
