from __future__ import annotations

from wdlm.data.trajectory import flatten_trajectory_steps, generate_trajectory_records


def test_generate_trajectories_produces_balanced_difficulties_and_unique_steps() -> None:
    trajectories = generate_trajectory_records(num_trajectories=6, seed=7)
    steps = flatten_trajectory_steps(trajectories)

    assert len(trajectories) == 6
    assert len({step.example_id for step in steps}) == len(steps)
    assert [trajectory.metadata.difficulty for trajectory in trajectories[:3]] == [
        "easy",
        "medium",
        "hard",
    ]
    assert all(3 <= trajectory.metadata.episode_length <= 12 for trajectory in trajectories)
    assert sum(trajectory.metadata.episode_length for trajectory in trajectories) == len(steps)

    easy_trajectory = trajectories[0]
    hard_trajectory = trajectories[2]
    assert len(easy_trajectory.initial_state.objects) == 3
    assert len(hard_trajectory.initial_state.objects) == 6
    assert all(step.action_struct.type != "give" for step in easy_trajectory.steps)
