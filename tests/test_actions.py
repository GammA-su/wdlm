from __future__ import annotations

import pytest

from wdlm.data.actions import InvalidTransitionError, apply_action
from wdlm.schemas import CloseAction, HideAction, MoveAction, ObjectState, RevealAction, WorldState


def make_state() -> WorldState:
    return WorldState(
        locations=["floor", "shelf", "table"],
        owners=["alice", "bob"],
        containers={"cabinet": "closed", "drawer": "open"},
        objects={
            "apple": ObjectState(holder="table", visibility="visible"),
            "blue_coin": ObjectState(holder="alice", visibility="hidden"),
            "book": ObjectState(holder="cabinet", visibility="hidden"),
            "red_key": ObjectState(holder="drawer", visibility="visible"),
        },
    )


def test_move_updates_holder() -> None:
    state = make_state()
    updated = apply_action(
        state,
        MoveAction(object="red_key", from_="drawer", to="table"),
    )
    assert updated.objects["red_key"].holder == "table"
    assert updated.objects["red_key"].visibility == "visible"


def test_close_hides_objects_inside_container() -> None:
    state = make_state()
    updated = apply_action(state, CloseAction(container="drawer"))
    assert updated.containers["drawer"] == "closed"
    assert updated.objects["red_key"].visibility == "hidden"


def test_invalid_move_from_closed_container_raises() -> None:
    state = make_state()
    with pytest.raises(InvalidTransitionError):
        apply_action(state, MoveAction(object="book", from_="cabinet", to="table"))


def test_invalid_reveal_inside_closed_container_raises() -> None:
    state = make_state()
    with pytest.raises(InvalidTransitionError):
        apply_action(state, RevealAction(object="book"))


def test_invalid_hide_of_hidden_object_raises() -> None:
    state = make_state()
    with pytest.raises(InvalidTransitionError):
        apply_action(state, HideAction(object="blue_coin"))
