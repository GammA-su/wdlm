"""Single-step dataset generation for the toy world benchmark."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from wdlm.data.actions import apply_action
from wdlm.data.toy_world import build_world_profile, canonicalize_state, generate_world_state
from wdlm.data.trajectory import build_step_record, list_profile_actions
from wdlm.schemas import ExampleRecord
from wdlm.utils.io import write_jsonl
from wdlm.utils.rng import SeededRNG


ProgressCallback = Callable[[dict[str, object]], None]
PARALLEL_EXAMPLE_THRESHOLD = 256


def _resolve_worker_count(total_tasks: int, requested_workers: int | None) -> int:
    if os.environ.get("WDLM_TEST_MODE") == "1" or "PYTEST_CURRENT_TEST" in os.environ:
        return 1
    cpu_count = os.process_cpu_count() or os.cpu_count() or 1
    if requested_workers is None or requested_workers <= 0:
        workers = min(16, cpu_count, max(1, total_tasks))
    else:
        workers = min(requested_workers, cpu_count, max(1, total_tasks))
    if total_tasks < PARALLEL_EXAMPLE_THRESHOLD:
        return 1
    return max(1, workers)


def _generate_single_example(index: int, seed: int) -> ExampleRecord:
    example_rng = SeededRNG(seed).derive("example", index)
    profile = build_world_profile(example_rng.derive("profile"), "medium")
    state_rng = example_rng.derive("state")
    action_rng = example_rng.derive("action")

    state_before = generate_world_state(state_rng, difficulty="medium", profile=profile)
    valid_actions = list_profile_actions(state_before, profile)
    action = action_rng.choice(valid_actions)
    state_after = canonicalize_state(apply_action(state_before, action))
    return build_step_record(
        state_before=state_before,
        action=action,
        state_after=state_after,
        world_id=f"world-{seed}-{index:06d}",
        trajectory_id=f"trajectory-{seed}-{index:06d}",
        step_index=0,
        episode_length=1,
        difficulty="medium",
        step_seed=example_rng.seed,
        allowed_actions=valid_actions,
    )


def generate_toy_world_examples(
    num_examples: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    workers: int | None = None,
) -> list[ExampleRecord]:
    """Generate deterministic single-step toy-world examples."""

    examples: list[ExampleRecord] = []
    progress_interval = max(1, min(250, num_examples // 20 or 1))
    resolved_workers = _resolve_worker_count(num_examples, workers)

    def _maybe_report(index: int, example: ExampleRecord) -> None:
        if progress_callback is not None and (
            index == 0
            or index + 1 == num_examples
            or (index + 1) % progress_interval == 0
        ):
            progress_callback(
                {
                    "phase": "generate",
                    "example_index": index,
                    "examples_completed": index + 1,
                    "num_examples": num_examples,
                    "workers": resolved_workers,
                    "example_id": example.example_id,
                    "action": example.action_struct.model_dump(mode="json", by_alias=True),
                    "state_before": example.state_before.model_dump(mode="json"),
                    "state_after": example.state_after.model_dump(mode="json"),
                }
            )

    if resolved_workers == 1:
        for index in range(num_examples):
            example = _generate_single_example(index, seed)
            examples.append(example)
            _maybe_report(index, example)
        return examples

    chunksize = max(1, num_examples // (resolved_workers * 8))
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        iterator = executor.map(
            _generate_single_example,
            range(num_examples),
            [seed] * num_examples,
            chunksize=chunksize,
        )
        for index, example in enumerate(iterator):
            examples.append(example)
            _maybe_report(index, example)
    return examples


def generate_toy_world_file(
    out: Path,
    num_examples: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    workers: int | None = None,
) -> None:
    """Generate a toy-world dataset and write it to JSONL."""

    resolved_workers = _resolve_worker_count(num_examples, workers)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "start",
                "num_examples": num_examples,
                "seed": seed,
                "workers": resolved_workers,
                "out": str(out),
            }
        )
    examples = generate_toy_world_examples(
        num_examples=num_examples,
        seed=seed,
        progress_callback=progress_callback,
        workers=resolved_workers,
    )
    rows = [example.model_dump(mode="json", by_alias=True) for example in examples]
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "write",
                "example_count": len(rows),
                "out": str(out),
            }
        )
    write_jsonl(out, rows)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "example_count": len(rows),
                "out": str(out),
            }
        )
