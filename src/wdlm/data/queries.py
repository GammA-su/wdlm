"""State-query generation for WDLM toy-world datasets."""

from __future__ import annotations

from pathlib import Path

from wdlm.data.renderer import humanize_name
from wdlm.data.toy_world import owner_for_object
from wdlm.schemas import ExampleRecord, QueryMetadata, QueryRecord
from wdlm.utils.io import read_jsonl, write_jsonl
from wdlm.utils.rng import derive_seed


def load_step_examples(path: Path) -> list[ExampleRecord]:
    """Load step examples from JSONL."""

    return [ExampleRecord.model_validate(row) for row in read_jsonl(path)]


def question_specs_for_example(example: ExampleRecord) -> list[tuple[str, str, str]]:
    """Return the deterministic question/answer specs for one example."""

    state = example.state_after
    object_names = sorted(state.objects)
    container_names = sorted(state.containers)

    where_object = object_names[example.step_index % len(object_names)]
    owner_object = object_names[(example.step_index + 1) % len(object_names)]
    visibility_object = object_names[(example.step_index + 2) % len(object_names)]
    container_name = container_names[example.step_index % len(container_names)]

    owner = owner_for_object(state, owner_object) or "nobody"
    container_is_open = "yes" if state.containers[container_name] == "open" else "no"
    object_is_visible = "yes" if state.objects[visibility_object].visibility == "visible" else "no"

    return [
        (
            "where_is_object",
            f"Where is the {humanize_name(where_object)}?",
            state.objects[where_object].holder,
        ),
        (
            "who_owns_object",
            f"Who owns the {humanize_name(owner_object)}?",
            owner,
        ),
        (
            "is_container_open",
            f"Is the {humanize_name(container_name)} open?",
            container_is_open,
        ),
        (
            "is_object_visible",
            f"Is the {humanize_name(visibility_object)} visible?",
            object_is_visible,
        ),
    ]


def generate_query_records(
    steps: list[ExampleRecord],
    *,
    seed: int,
) -> list[QueryRecord]:
    """Generate deterministic QA records from per-step examples."""

    queries: list[QueryRecord] = []
    for example in steps:
        for question_type, question, answer in question_specs_for_example(example):
            queries.append(
                QueryRecord(
                    qa_id=f"{example.example_id}--{question_type}",
                    example_id=example.example_id,
                    world_id=example.world_id,
                    trajectory_id=example.trajectory_id,
                    step_index=example.step_index,
                    question_type=question_type,
                    question=question,
                    answer=answer,
                    metadata=QueryMetadata(
                        split=example.metadata.split,
                        difficulty=example.metadata.difficulty,
                        seed=derive_seed(seed, example.example_id, question_type),
                    ),
                )
            )
    return queries


def generate_query_file(
    *,
    input_steps_path: Path,
    out_path: Path,
    seed: int,
) -> Path:
    """Generate a QA JSONL file from per-step examples."""

    steps = load_step_examples(input_steps_path)
    queries = generate_query_records(steps, seed=seed)
    write_jsonl(out_path, [query.model_dump(mode="json") for query in queries])
    return out_path
