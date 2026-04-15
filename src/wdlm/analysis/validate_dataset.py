"""Dataset invariant validation for WDLM Deliverable B."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wdlm.data.actions import apply_action, filter_distinct_actions
from wdlm.data.ood import LENGTH_OOD_THRESHOLD, build_ood_splits, composition_signature
from wdlm.data.paraphrase import generate_paraphrase_entries
from wdlm.data.queries import question_specs_for_example
from wdlm.data.renderer import TEMPLATES, render_action
from wdlm.data.split import load_examples
from wdlm.data.trajectory import generate_trajectory_records
from wdlm.schemas import ExampleRecord, QueryRecord, TrajectoryRecord
from wdlm.utils.io import read_jsonl


EXPECTED_OOD_SPLITS: tuple[str, ...] = (
    "train",
    "val",
    "test_iid",
    "test_lexical_ood",
    "test_compositional_ood",
    "test_length_ood",
    "test_paraphrase_ood",
)


class ValidationReport(BaseModel):
    """Structured validation report for dataset artifacts."""

    model_config = ConfigDict(extra="forbid")

    step_count: int
    trajectory_count: int
    qa_count: int
    split_counts: dict[str, int]
    leakage_report: dict[str, dict[str, int]]
    checks: list[str]


class DatasetValidationError(ValueError):
    """Raised when a dataset artifact fails invariant validation."""


def _load_trajectories(path: Path) -> list[TrajectoryRecord]:
    return [TrajectoryRecord.model_validate(row) for row in read_jsonl(path)]


def _load_queries(path: Path) -> list[QueryRecord]:
    return [QueryRecord.model_validate(row) for row in read_jsonl(path)]


def _template_index(template_id: str, action_type: str) -> int:
    prefix = f"{action_type}_t"
    if not template_id.startswith(prefix):
        raise DatasetValidationError(
            f"Template id {template_id} does not match action type {action_type}."
        )
    try:
        index = int(template_id[len(prefix) :])
    except ValueError as exc:
        raise DatasetValidationError(f"Invalid template id: {template_id}") from exc
    if action_type not in TEMPLATES or not (0 <= index < len(TEMPLATES[action_type])):
        raise DatasetValidationError(f"Template index out of range: {template_id}")
    return index


def _expected_paraphrases(example: ExampleRecord) -> list[tuple[str, str]]:
    chosen_index = _template_index(example.metadata.template_id, example.action_struct.type)
    return generate_paraphrase_entries(example.action_struct, exclude_template=chosen_index)


def _validate_steps(steps: list[ExampleRecord]) -> list[str]:
    seen_example_ids: set[str] = set()
    checks = ["step schema validity"]
    for example in steps:
        if example.example_id in seen_example_ids:
            raise DatasetValidationError(f"Duplicate example_id: {example.example_id}")
        seen_example_ids.add(example.example_id)

        expected_state_after = apply_action(example.state_before, example.action_struct)
        if expected_state_after.model_dump(mode="json") != example.state_after.model_dump(mode="json"):
            raise DatasetValidationError(
                f"state_after does not match applied action for {example.example_id}"
            )

        chosen_template_index = _template_index(
            example.metadata.template_id,
            example.action_struct.type,
        )
        expected_text = render_action(example.action_struct, template_index=chosen_template_index)
        if example.text_chunk != expected_text:
            raise DatasetValidationError(f"text_chunk mismatch for {example.example_id}")

        expected_paraphrases = _expected_paraphrases(example)
        if expected_paraphrases != list(
            zip(example.metadata.paraphrase_template_ids, example.paraphrases, strict=True)
        ):
            raise DatasetValidationError(f"Paraphrase template/text mismatch for {example.example_id}")

        distinct_negatives = filter_distinct_actions(
            example.state_before,
            example.action_struct,
            [negative.action_struct for negative in example.negative_updates],
        )
        if len(distinct_negatives) != len(example.negative_updates):
            raise DatasetValidationError(
                f"Negative update semantics collide with positive action for {example.example_id}"
            )
        for negative in example.negative_updates:
            negative_template_index = _template_index(
                negative.template_id,
                negative.action_struct.type,
            )
            expected_negative_text = render_action(
                negative.action_struct,
                template_index=negative_template_index,
            )
            if negative.text_chunk != expected_negative_text:
                raise DatasetValidationError(
                    f"Negative text/template mismatch for {example.example_id}"
                )
    checks.extend(
        [
            "unique example_id",
            "state_after equals apply_action(state_before, action_struct)",
            "text_chunk and paraphrase templates render deterministically",
            "negative updates remain semantically distinct",
        ]
    )
    return checks


def _validate_trajectories(
    steps: list[ExampleRecord],
    trajectories: list[TrajectoryRecord],
) -> list[str]:
    seen_trajectory_ids: set[str] = set()
    step_map = {step.example_id: step for step in steps}
    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise DatasetValidationError(f"Duplicate trajectory_id: {trajectory.trajectory_id}")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        if trajectory.initial_state.model_dump(mode="json") != trajectory.steps[0].state_before.model_dump(
            mode="json"
        ):
            raise DatasetValidationError(
                f"Initial state mismatch for trajectory {trajectory.trajectory_id}"
            )
        if trajectory.final_state.model_dump(mode="json") != trajectory.steps[-1].state_after.model_dump(
            mode="json"
        ):
            raise DatasetValidationError(
                f"Final state mismatch for trajectory {trajectory.trajectory_id}"
            )
        for index, step in enumerate(trajectory.steps):
            if step.example_id not in step_map:
                raise DatasetValidationError(
                    f"Trajectory step {step.example_id} missing from step dataset."
                )
            if step_map[step.example_id].model_dump(mode="json") != step.model_dump(mode="json"):
                raise DatasetValidationError(
                    f"Trajectory step payload mismatch for {step.example_id}."
                )
            if index > 0:
                previous = trajectory.steps[index - 1]
                if previous.state_after.model_dump(mode="json") != step.state_before.model_dump(mode="json"):
                    raise DatasetValidationError(
                        f"Broken trajectory state chain for {trajectory.trajectory_id}"
                    )
    return [
        "trajectory schema validity",
        "unique trajectory_id in trajectory file",
        "trajectory steps align with flattened step dataset",
    ]


def _validate_queries(
    steps: list[ExampleRecord],
    queries: list[QueryRecord],
) -> list[str]:
    seen_qa_ids: set[str] = set()
    step_map = {step.example_id: step for step in steps}
    for query in queries:
        if query.qa_id in seen_qa_ids:
            raise DatasetValidationError(f"Duplicate qa_id: {query.qa_id}")
        seen_qa_ids.add(query.qa_id)
        if query.example_id not in step_map:
            raise DatasetValidationError(f"QA references unknown example_id: {query.example_id}")
        example = step_map[query.example_id]
        expected_specs = {
            question_type: (question, answer)
            for question_type, question, answer in question_specs_for_example(example)
        }
        expected_question, expected_answer = expected_specs[query.question_type]
        if query.question != expected_question or query.answer != expected_answer:
            raise DatasetValidationError(f"QA content mismatch for {query.qa_id}")
        if query.world_id != example.world_id or query.trajectory_id != example.trajectory_id:
            raise DatasetValidationError(f"QA linkage mismatch for {query.qa_id}")
    return [
        "QA schema validity",
        "QA links resolve to generated steps",
        "QA question/answer content matches state-derived expectations",
    ]


def _check_expected_split_contents(
    actual_splits: dict[str, list[ExampleRecord]],
    expected_splits: dict[str, list[ExampleRecord]],
) -> None:
    for split_name in EXPECTED_OOD_SPLITS:
        actual_ids = [example.example_id for example in actual_splits.get(split_name, [])]
        expected_ids = [example.example_id for example in expected_splits.get(split_name, [])]
        if actual_ids != expected_ids:
            raise DatasetValidationError(
                f"Split membership mismatch for {split_name}. "
                f"Expected {len(expected_ids)} ids, found {len(actual_ids)} ids."
            )


def _compute_leakage_report(
    train_examples: list[ExampleRecord],
    lexical_holdouts: set[str],
    paraphrase_holdouts: set[str],
    compositional_holdouts: set[str],
) -> dict[str, dict[str, int]]:
    lexical_violations = sum(
        1
        for example in train_examples
        if example.metadata.template_id in lexical_holdouts
    )
    paraphrase_surface_violations = sum(
        1
        for example in train_examples
        if set(example.metadata.paraphrase_template_ids) & paraphrase_holdouts
    )
    compositional_violations = sum(
        1
        for example in train_examples
        if composition_signature(example) in compositional_holdouts
    )
    length_violations = sum(
        1
        for example in train_examples
        if example.episode_length > LENGTH_OOD_THRESHOLD
    )
    return {
        "lexical_ood": {"train_exposure_violations": lexical_violations},
        "paraphrase_ood": {"train_exposure_violations": paraphrase_surface_violations},
        "compositional_ood": {"train_exposure_violations": compositional_violations},
        "length_ood": {"train_exposure_violations": length_violations},
    }


def _validate_splits(
    steps: list[ExampleRecord],
    split_dir: Path,
    *,
    seed: int,
) -> tuple[list[str], dict[str, int], dict[str, dict[str, int]]]:
    actual_splits: dict[str, list[ExampleRecord]] = {}
    for split_name in EXPECTED_OOD_SPLITS:
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise DatasetValidationError(f"Missing split file: {path}")
        actual_splits[split_name] = load_examples(path)

    expected_splits, summary = build_ood_splits(steps, seed=seed)
    _check_expected_split_contents(actual_splits, expected_splits)

    lexical_holdouts = set(summary.lexical_holdout_template_ids)
    paraphrase_holdouts = set(summary.paraphrase_holdout_template_ids)
    compositional_holdouts = set(summary.compositional_holdouts)

    leakage_report = _compute_leakage_report(
        actual_splits["train"],
        lexical_holdouts=lexical_holdouts,
        paraphrase_holdouts=paraphrase_holdouts,
        compositional_holdouts=compositional_holdouts,
    )
    if any(
        bucket["train_exposure_violations"] > 0
        for bucket in leakage_report.values()
    ):
        raise DatasetValidationError(f"OOD train leakage detected: {leakage_report}")

    for example in actual_splits["test_lexical_ood"]:
        if example.metadata.template_id not in lexical_holdouts:
            raise DatasetValidationError(
                f"Lexical OOD example missing lexical holdout template: {example.example_id}"
            )
    for example in actual_splits["test_compositional_ood"]:
        if composition_signature(example) not in compositional_holdouts:
            raise DatasetValidationError(
                f"Compositional OOD example missing compositional holdout: {example.example_id}"
            )
    for example in actual_splits["test_length_ood"]:
        if example.episode_length <= LENGTH_OOD_THRESHOLD:
            raise DatasetValidationError(
                f"Length OOD example below threshold: {example.example_id}"
            )
    for example in actual_splits["test_paraphrase_ood"]:
        if example.metadata.template_id not in paraphrase_holdouts:
            raise DatasetValidationError(
                f"Paraphrase OOD example missing paraphrase holdout template: {example.example_id}"
            )

    split_counts = {
        split_name: len(split_examples)
        for split_name, split_examples in actual_splits.items()
    }
    checks = [
        "OOD split files exist",
        "OOD split membership exactly matches deterministic builder",
        "train leakage report is clean for lexical/paraphrase/compositional/length OOD",
    ]
    return checks, split_counts, leakage_report


def validate_dataset_artifacts(
    *,
    steps_path: Path,
    trajectories_path: Path | None = None,
    qa_path: Path | None = None,
    split_dir: Path | None = None,
    seed: int = 42,
) -> ValidationReport:
    """Validate WDLM Deliverable B artifacts and return a structured report."""

    steps = load_examples(steps_path)
    checks = _validate_steps(steps)

    trajectories: list[TrajectoryRecord] = []
    if trajectories_path is not None:
        trajectories = _load_trajectories(trajectories_path)
        checks.extend(_validate_trajectories(steps, trajectories))

    queries: list[QueryRecord] = []
    if qa_path is not None:
        queries = _load_queries(qa_path)
        checks.extend(_validate_queries(steps, queries))

    split_counts: dict[str, int] = {}
    leakage_report: dict[str, dict[str, int]] = {}
    if split_dir is not None:
        split_checks, split_counts, leakage_report = _validate_splits(
            steps,
            split_dir,
            seed=seed,
        )
        checks.extend(split_checks)

    return ValidationReport(
        step_count=len(steps),
        trajectory_count=len(trajectories),
        qa_count=len(queries),
        split_counts=split_counts,
        leakage_report=leakage_report,
        checks=checks,
    )
