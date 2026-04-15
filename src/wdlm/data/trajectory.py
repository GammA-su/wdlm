"""Trajectory generation for the WDLM toy-world benchmark."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from wdlm.data.actions import apply_action, list_valid_actions
from wdlm.data.negatives import generate_negative_updates
from wdlm.data.paraphrase import generate_paraphrase_entries
from wdlm.data.renderer import render_action, template_count, template_id_for_action
from wdlm.data.toy_world import (
    WorldProfile,
    build_world_profile,
    canonicalize_state,
    episode_length_for_profile,
    generate_world_state,
)
from wdlm.schemas import ExampleMetadata, ExampleRecord, TrajectoryMetadata, TrajectoryRecord
from wdlm.utils.io import write_jsonl
from wdlm.utils.rng import SeededRNG


DIFFICULTY_SEQUENCE: tuple[str, ...] = ("easy", "medium", "hard")
ProgressCallback = Callable[[dict[str, object]], None]
PARALLEL_TRAJECTORY_THRESHOLD = 256


def choose_difficulty(trajectory_index: int) -> str:
    """Choose a deterministic difficulty tier for a trajectory index."""

    return DIFFICULTY_SEQUENCE[trajectory_index % len(DIFFICULTY_SEQUENCE)]


def list_profile_actions(state, profile: WorldProfile):
    """List valid actions allowed under the active world profile."""

    actions = list_valid_actions(
        state,
        allowed_action_types=set(profile.allowed_action_types),
    )
    filtered = []
    for action in actions:
        if action.type == "move" and not profile.allow_owner_holders:
            if action.from_ in profile.owners or action.to in profile.owners:
                continue
        filtered.append(action)
    return filtered


def build_step_record(
    *,
    state_before,
    action,
    state_after,
    world_id: str,
    trajectory_id: str,
    step_index: int,
    episode_length: int,
    difficulty: str,
    step_seed: int,
    allowed_actions,
) -> ExampleRecord:
    """Build a typed step record from one transition."""

    text_rng = SeededRNG(step_seed).derive("text")
    chosen_template = text_rng.randint(0, template_count(action) - 1)
    text_chunk = render_action(action, template_index=chosen_template)
    paraphrase_entries = generate_paraphrase_entries(action, exclude_template=chosen_template)
    negatives = generate_negative_updates(
        state_before,
        action,
        allowed_action_types={candidate.type for candidate in allowed_actions},
        candidate_actions=allowed_actions,
    )
    return ExampleRecord(
        example_id=f"example-{trajectory_id}-{step_index:02d}",
        world_id=world_id,
        trajectory_id=trajectory_id,
        step_index=step_index,
        episode_length=episode_length,
        state_before=state_before,
        action_struct=action,
        text_chunk=text_chunk,
        paraphrases=[text for _, text in paraphrase_entries],
        negative_updates=negatives,
        state_after=state_after,
        metadata=ExampleMetadata(
            split="unsplit",
            template_id=template_id_for_action(action, chosen_template),
            paraphrase_template_ids=[template_id for template_id, _ in paraphrase_entries],
            difficulty=difficulty,
            seed=step_seed,
        ),
    )


def _resolve_worker_count(total_tasks: int, requested_workers: int | None) -> int:
    if os.environ.get("WDLM_TEST_MODE") == "1" or "PYTEST_CURRENT_TEST" in os.environ:
        return 1
    cpu_count = os.process_cpu_count() or os.cpu_count() or 1
    if requested_workers is None or requested_workers <= 0:
        workers = min(16, cpu_count, max(1, total_tasks))
    else:
        workers = min(requested_workers, cpu_count, max(1, total_tasks))
    if total_tasks < PARALLEL_TRAJECTORY_THRESHOLD:
        return 1
    return max(1, workers)


def _generate_single_trajectory(trajectory_index: int, seed: int) -> TrajectoryRecord:
    trajectory_rng = SeededRNG(seed).derive("trajectory", trajectory_index)
    difficulty = choose_difficulty(trajectory_index)
    profile = build_world_profile(trajectory_rng.derive("profile"), difficulty)
    initial_state = generate_world_state(
        trajectory_rng.derive("initial-state"),
        difficulty=difficulty,
        profile=profile,
    )
    episode_length = episode_length_for_profile(
        trajectory_rng.derive("length"),
        profile,
    )
    world_id = f"world-{seed}-{trajectory_index:06d}"
    trajectory_id = f"trajectory-{seed}-{trajectory_index:06d}"

    current_state = initial_state
    steps: list[ExampleRecord] = []
    for step_index in range(episode_length):
        step_rng = trajectory_rng.derive("step", step_index)
        allowed_actions = list_profile_actions(current_state, profile)
        if not allowed_actions:
            raise RuntimeError(
                f"No valid actions available for {trajectory_id} at step {step_index}."
            )
        action = step_rng.derive("action").choice(allowed_actions)
        state_before = canonicalize_state(current_state)
        state_after = canonicalize_state(apply_action(state_before, action))
        step_record = build_step_record(
            state_before=state_before,
            action=action,
            state_after=state_after,
            world_id=world_id,
            trajectory_id=trajectory_id,
            step_index=step_index,
            episode_length=episode_length,
            difficulty=difficulty,
            step_seed=step_rng.seed,
            allowed_actions=allowed_actions,
        )
        steps.append(step_record)
        current_state = state_after

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        world_id=world_id,
        initial_state=initial_state,
        final_state=current_state,
        steps=steps,
        metadata=TrajectoryMetadata(
            split="unsplit",
            difficulty=difficulty,
            seed=trajectory_rng.seed,
            episode_length=episode_length,
        ),
    )


def generate_trajectory_records(
    *,
    num_trajectories: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    workers: int | None = None,
) -> list[TrajectoryRecord]:
    """Generate deterministic multi-step trajectories across difficulty tiers."""

    trajectories: list[TrajectoryRecord] = []
    total_steps = 0
    progress_interval = max(1, min(250, num_trajectories // 20 or 1))
    resolved_workers = _resolve_worker_count(num_trajectories, workers)

    def _maybe_report(trajectory_index: int, trajectory: TrajectoryRecord) -> None:
        nonlocal total_steps
        total_steps += trajectory.metadata.episode_length
        if progress_callback is not None and (
            trajectory_index == 0
            or trajectory_index + 1 == num_trajectories
            or (trajectory_index + 1) % progress_interval == 0
        ):
            progress_callback(
                {
                    "phase": "generate",
                    "trajectory_index": trajectory_index,
                    "trajectories_completed": trajectory_index + 1,
                    "num_trajectories": num_trajectories,
                    "workers": resolved_workers,
                    "trajectory_id": trajectory.trajectory_id,
                    "difficulty": trajectory.metadata.difficulty,
                    "episode_length": trajectory.metadata.episode_length,
                    "steps_completed": total_steps,
                    "final_state": trajectory.final_state.model_dump(mode="json"),
                }
            )

    if resolved_workers == 1:
        for trajectory_index in range(num_trajectories):
            trajectory = _generate_single_trajectory(trajectory_index, seed)
            trajectories.append(trajectory)
            _maybe_report(trajectory_index, trajectory)
        return trajectories

    chunksize = max(1, num_trajectories // (resolved_workers * 8))
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        iterator = executor.map(
            _generate_single_trajectory,
            range(num_trajectories),
            [seed] * num_trajectories,
            chunksize=chunksize,
        )
        for trajectory_index, trajectory in enumerate(iterator):
            trajectories.append(trajectory)
            _maybe_report(trajectory_index, trajectory)

    return trajectories


def flatten_trajectory_steps(trajectories: list[TrajectoryRecord]) -> list[ExampleRecord]:
    """Flatten trajectories into a single per-step dataset."""

    steps: list[ExampleRecord] = []
    for trajectory in trajectories:
        steps.extend(trajectory.steps)
    return steps


def generate_trajectory_files(
    *,
    out_steps: Path,
    out_trajectories: Path,
    num_trajectories: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    workers: int | None = None,
) -> tuple[Path, Path]:
    """Generate trajectory and per-step JSONL outputs."""

    resolved_workers = _resolve_worker_count(num_trajectories, workers)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "start",
                "num_trajectories": num_trajectories,
                "seed": seed,
                "workers": resolved_workers,
                "out_steps": str(out_steps),
                "out_trajectories": str(out_trajectories),
            }
        )
    trajectories = generate_trajectory_records(
        num_trajectories=num_trajectories,
        seed=seed,
        progress_callback=progress_callback,
        workers=resolved_workers,
    )
    steps = flatten_trajectory_steps(trajectories)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "write_steps",
                "step_count": len(steps),
                "out_steps": str(out_steps),
            }
        )
    write_jsonl(
        out_steps,
        [step.model_dump(mode="json", by_alias=True) for step in steps],
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "write_trajectories",
                "trajectory_count": len(trajectories),
                "out_trajectories": str(out_trajectories),
            }
        )
    write_jsonl(
        out_trajectories,
        [trajectory.model_dump(mode="json", by_alias=True) for trajectory in trajectories],
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "step_count": len(steps),
                "trajectory_count": len(trajectories),
                "out_steps": str(out_steps),
                "out_trajectories": str(out_trajectories),
            }
        )
    return out_steps, out_trajectories
