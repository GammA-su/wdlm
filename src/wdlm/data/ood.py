"""OOD split construction for WDLM toy-world datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wdlm.data.split import assign_split, load_examples, split_examples
from wdlm.schemas import ExampleMetadata, ExampleRecord, GiveAction, MoveAction
from wdlm.utils.io import write_jsonl
from wdlm.utils.rng import SeededRNG, derive_seed


LENGTH_OOD_THRESHOLD = 8


@dataclass(frozen=True)
class OODSplitSummary:
    """Summary information for one OOD split build."""

    lexical_holdout_template_ids: tuple[str, ...]
    paraphrase_holdout_template_ids: tuple[str, ...]
    compositional_holdouts: tuple[str, ...]
    length_threshold: int
    split_sizes: dict[str, int]


def _template_ids_by_action_type(examples: list[ExampleRecord]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for example in examples:
        grouped.setdefault(example.action_struct.type, set()).add(example.metadata.template_id)
        grouped[example.action_struct.type].update(example.metadata.paraphrase_template_ids)
    return {
        action_type: sorted(template_ids)
        for action_type, template_ids in grouped.items()
    }


def _select_template_holdouts(
    examples: list[ExampleRecord],
    *,
    seed: int,
) -> tuple[set[str], set[str]]:
    template_ids_by_type = _template_ids_by_action_type(examples)
    lexical_holdouts: set[str] = set()
    paraphrase_holdouts: set[str] = set()
    for action_type, template_ids in sorted(template_ids_by_type.items()):
        lexical_index = derive_seed(seed, "lexical", action_type) % len(template_ids)
        lexical_template = template_ids[lexical_index]
        lexical_holdouts.add(lexical_template)

        remaining = [template_id for template_id in template_ids if template_id != lexical_template]
        paraphrase_index = derive_seed(seed, "paraphrase", action_type) % len(remaining)
        paraphrase_holdouts.add(remaining[paraphrase_index])
    return lexical_holdouts, paraphrase_holdouts


def composition_signature(example: ExampleRecord) -> str:
    """Return a deterministic composition signature for compositional OOD."""

    action = example.action_struct
    if isinstance(action, MoveAction):
        return f"move|{action.object}|{action.from_}|{action.to}"
    if isinstance(action, GiveAction):
        return f"give|{action.object}|{action.from_owner}|{action.to_owner}"
    if hasattr(action, "container"):
        return f"{action.type}|{action.container}"
    object_holder = example.state_before.objects[action.object].holder
    return f"{action.type}|{action.object}|{object_holder}"


def _select_compositional_holdouts(
    examples: list[ExampleRecord],
    *,
    seed: int,
) -> set[str]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        grouped.setdefault(example.action_struct.type, []).append(composition_signature(example))
    holdouts: set[str] = set()
    for action_type, signatures in sorted(grouped.items()):
        unique_signatures = sorted(set(signatures))
        chosen_index = derive_seed(seed, "compositional", action_type) % len(unique_signatures)
        holdouts.add(unique_signatures[chosen_index])
    return holdouts


def _sanitize_for_training(
    example: ExampleRecord,
    *,
    blocked_template_ids: set[str],
) -> ExampleRecord:
    pairs = [
        (template_id, text)
        for template_id, text in zip(
            example.metadata.paraphrase_template_ids,
            example.paraphrases,
            strict=True,
        )
        if template_id not in blocked_template_ids
    ]
    if not pairs:
        raise ValueError(
            f"Example {example.example_id} has no remaining paraphrases after OOD filtering."
        )
    return example.model_copy(
        update={
            "paraphrases": [text for _, text in pairs],
            "metadata": ExampleMetadata(
                split=example.metadata.split,
                template_id=example.metadata.template_id,
                paraphrase_template_ids=[template_id for template_id, _ in pairs],
                difficulty=example.metadata.difficulty,
                seed=example.metadata.seed,
            ),
        }
    )


def _build_paraphrase_ood_clone(
    example: ExampleRecord,
    *,
    paraphrase_holdouts: set[str],
    blocked_template_ids: set[str],
) -> ExampleRecord | None:
    heldout_pairs = [
        (template_id, text)
        for template_id, text in zip(
            example.metadata.paraphrase_template_ids,
            example.paraphrases,
            strict=True,
        )
        if template_id in paraphrase_holdouts
    ]
    if not heldout_pairs:
        return None

    heldout_template_id, heldout_text = heldout_pairs[0]
    remaining_pairs = [(example.metadata.template_id, example.text_chunk)] + [
        (template_id, text)
        for template_id, text in zip(
            example.metadata.paraphrase_template_ids,
            example.paraphrases,
            strict=True,
        )
        if template_id not in blocked_template_ids and template_id != heldout_template_id
    ]
    return example.model_copy(
        update={
            "example_id": f"{example.example_id}--paraphrase-ood",
            "text_chunk": heldout_text,
            "paraphrases": [text for _, text in remaining_pairs],
            "metadata": ExampleMetadata(
                split="test_paraphrase_ood",
                template_id=heldout_template_id,
                paraphrase_template_ids=[template_id for template_id, _ in remaining_pairs],
                difficulty=example.metadata.difficulty,
                seed=derive_seed(example.metadata.seed, "paraphrase-ood"),
            ),
        }
    )


def build_ood_splits(
    examples: list[ExampleRecord],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[dict[str, list[ExampleRecord]], OODSplitSummary]:
    """Build IID and OOD evaluation splits from per-step examples."""

    lexical_holdouts, paraphrase_holdouts = _select_template_holdouts(examples, seed=seed)
    compositional_holdouts = _select_compositional_holdouts(examples, seed=seed)
    blocked_template_ids = lexical_holdouts | paraphrase_holdouts

    lexical_examples: list[ExampleRecord] = []
    compositional_examples: list[ExampleRecord] = []
    length_examples: list[ExampleRecord] = []
    paraphrase_main_examples: list[ExampleRecord] = []
    base_examples: list[ExampleRecord] = []
    paraphrase_clone_sources: list[ExampleRecord] = []

    for example in sorted(examples, key=lambda item: item.example_id):
        if example.episode_length > LENGTH_OOD_THRESHOLD:
            length_examples.append(assign_split(example, "test_length_ood"))
            continue
        if composition_signature(example) in compositional_holdouts:
            compositional_examples.append(assign_split(example, "test_compositional_ood"))
            continue
        if example.metadata.template_id in lexical_holdouts:
            lexical_examples.append(assign_split(example, "test_lexical_ood"))
            continue
        if example.metadata.template_id in paraphrase_holdouts:
            paraphrase_main_examples.append(assign_split(example, "test_paraphrase_ood"))
            continue
        paraphrase_clone_sources.append(example)
        sanitized = _sanitize_for_training(example, blocked_template_ids=blocked_template_ids)
        base_examples.append(sanitized)

    iid_splits = split_examples(
        base_examples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    paraphrase_clone_examples = [
        clone
        for clone in (
            _build_paraphrase_ood_clone(
                example,
                paraphrase_holdouts=paraphrase_holdouts,
                blocked_template_ids=blocked_template_ids,
            )
            for example in paraphrase_clone_sources
        )
        if clone is not None
    ]

    splits: dict[str, list[ExampleRecord]] = {
        "train": iid_splits["train"],
        "val": iid_splits["val"],
        "test_iid": [assign_split(example, "test_iid") for example in iid_splits["test"]],
        "test_lexical_ood": lexical_examples,
        "test_compositional_ood": compositional_examples,
        "test_length_ood": length_examples,
        "test_paraphrase_ood": paraphrase_main_examples
        + paraphrase_clone_examples,
    }

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda example: example.example_id)

    summary = OODSplitSummary(
        lexical_holdout_template_ids=tuple(sorted(lexical_holdouts)),
        paraphrase_holdout_template_ids=tuple(sorted(paraphrase_holdouts)),
        compositional_holdouts=tuple(sorted(compositional_holdouts)),
        length_threshold=LENGTH_OOD_THRESHOLD,
        split_sizes={split_name: len(split_examples) for split_name, split_examples in splits.items()},
    )
    return splits, summary


def build_ood_splits_file(
    *,
    input_path: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[dict[str, Path], OODSplitSummary]:
    """Read step examples, build OOD splits, and write JSONL outputs."""

    examples = load_examples(input_path)
    splits, summary = build_ood_splits(
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
    return paths, summary
