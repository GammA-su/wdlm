from __future__ import annotations

from wdlm.data.actions import apply_action
from wdlm.data.generate import generate_toy_world_examples
from wdlm.data.negatives import generate_negative_updates
from wdlm.schemas import ExampleRecord, MoveAction, ObjectState, WorldState


def make_state() -> WorldState:
    return WorldState(
        locations=["floor", "shelf", "table"],
        owners=["alice", "bob"],
        containers={"cabinet": "closed", "drawer": "open"},
        objects={
            "apple": ObjectState(holder="table", visibility="visible"),
            "blue_coin": ObjectState(holder="drawer", visibility="visible"),
            "book": ObjectState(holder="alice", visibility="hidden"),
            "red_key": ObjectState(holder="shelf", visibility="visible"),
        },
    )


def test_negative_updates_are_semantically_distinct() -> None:
    state = make_state()
    positive = MoveAction(object="blue_coin", from_="drawer", to="table")
    positive_after = apply_action(state, positive)
    negatives = generate_negative_updates(state, positive)

    assert negatives
    for negative in negatives:
        negative_after = apply_action(state, negative.action_struct)
        assert negative_after.model_dump(mode="json") != positive_after.model_dump(mode="json")
        assert negative.text_chunk != ""
        assert negative.negative_type in {
            "wrong_object",
            "wrong_destination",
            "opposite_operation",
            "wrong_owner",
            "fallback",
        }


def test_generated_examples_are_schema_valid() -> None:
    examples = generate_toy_world_examples(num_examples=8, seed=42)
    assert len(examples) == 8
    assert len({example.example_id for example in examples}) == 8

    for example in examples:
        payload = example.model_dump(mode="json", by_alias=True)
        reparsed = ExampleRecord.model_validate(payload)
        assert reparsed.metadata.split == "unsplit"
        assert reparsed.paraphrases
        assert reparsed.negative_updates
        assert reparsed.step_index == 0
        assert reparsed.episode_length == 1
