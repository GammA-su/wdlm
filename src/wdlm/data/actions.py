"""Transition engine and valid-action enumeration."""

from __future__ import annotations

from collections.abc import Iterable

from wdlm.data.toy_world import (
    all_holders,
    canonicalize_state,
    is_container,
    is_holder,
    is_owner,
)
from wdlm.schemas import (
    ActionStruct,
    CloseAction,
    GiveAction,
    HideAction,
    MoveAction,
    ObjectState,
    OpenAction,
    RevealAction,
    WorldState,
    parse_action_struct,
)


class InvalidTransitionError(ValueError):
    """Raised when an action cannot be applied to a state."""


def _require_known_holder(state: WorldState, holder: str) -> None:
    if not is_holder(state, holder):
        raise InvalidTransitionError(f"Unknown holder: {holder}")


def _is_closed_container(state: WorldState, holder: str) -> bool:
    return is_container(state, holder) and state.containers[holder] == "closed"


def _get_object_state(state: WorldState, object_name: str) -> ObjectState:
    try:
        return state.objects[object_name]
    except KeyError as exc:
        raise InvalidTransitionError(f"Unknown object: {object_name}") from exc


def apply_action(state: WorldState, action: ActionStruct) -> WorldState:
    """Apply a validated action to a state and return the next state."""

    action = parse_action_struct(action)
    current_state = canonicalize_state(state)
    next_state = current_state.model_copy(deep=True)

    if isinstance(action, MoveAction):
        _require_known_holder(current_state, action.from_)
        _require_known_holder(current_state, action.to)
        object_state = _get_object_state(current_state, action.object)
        if object_state.holder != action.from_:
            raise InvalidTransitionError(
                f"Object {action.object} is at {object_state.holder}, not {action.from_}."
            )
        if action.from_ == action.to:
            raise InvalidTransitionError("Move source and destination must differ.")
        if is_owner(current_state, action.from_) and is_owner(current_state, action.to):
            raise InvalidTransitionError("Owner-to-owner transfers must use give.")
        if _is_closed_container(current_state, action.from_):
            raise InvalidTransitionError(
                f"Cannot move {action.object} out of closed container {action.from_}."
            )
        if _is_closed_container(current_state, action.to):
            raise InvalidTransitionError(
                f"Cannot move {action.object} into closed container {action.to}."
            )
        next_state.objects[action.object].holder = action.to
        return canonicalize_state(next_state)

    if isinstance(action, OpenAction):
        if action.container not in current_state.containers:
            raise InvalidTransitionError(f"Unknown container: {action.container}")
        if current_state.containers[action.container] == "open":
            raise InvalidTransitionError(f"Container {action.container} is already open.")
        next_state.containers[action.container] = "open"
        return canonicalize_state(next_state)

    if isinstance(action, CloseAction):
        if action.container not in current_state.containers:
            raise InvalidTransitionError(f"Unknown container: {action.container}")
        if current_state.containers[action.container] == "closed":
            raise InvalidTransitionError(f"Container {action.container} is already closed.")
        next_state.containers[action.container] = "closed"
        for object_name, object_state in next_state.objects.items():
            if object_state.holder == action.container:
                next_state.objects[object_name].visibility = "hidden"
        return canonicalize_state(next_state)

    if isinstance(action, GiveAction):
        if action.from_owner not in current_state.owners or action.to_owner not in current_state.owners:
            raise InvalidTransitionError("Give actions require known owners.")
        if action.from_owner == action.to_owner:
            raise InvalidTransitionError("Give source and target owners must differ.")
        object_state = _get_object_state(current_state, action.object)
        if object_state.holder != action.from_owner:
            raise InvalidTransitionError(
                f"Object {action.object} is held by {object_state.holder}, not {action.from_owner}."
            )
        next_state.objects[action.object].holder = action.to_owner
        return canonicalize_state(next_state)

    if isinstance(action, HideAction):
        object_state = _get_object_state(current_state, action.object)
        if object_state.visibility == "hidden":
            raise InvalidTransitionError(f"Object {action.object} is already hidden.")
        next_state.objects[action.object].visibility = "hidden"
        return canonicalize_state(next_state)

    if isinstance(action, RevealAction):
        object_state = _get_object_state(current_state, action.object)
        if object_state.visibility == "visible":
            raise InvalidTransitionError(f"Object {action.object} is already visible.")
        if _is_closed_container(current_state, object_state.holder):
            raise InvalidTransitionError(
                f"Cannot reveal {action.object} while it is inside closed container {object_state.holder}."
            )
        next_state.objects[action.object].visibility = "visible"
        return canonicalize_state(next_state)

    raise InvalidTransitionError(f"Unsupported action type: {action.type}")


def list_valid_actions(
    state: WorldState,
    *,
    allowed_action_types: set[str] | None = None,
) -> list[ActionStruct]:
    """Enumerate all valid actions for a state in deterministic order."""

    current_state = canonicalize_state(state)
    allowed = allowed_action_types
    actions: list[ActionStruct] = []

    for container in current_state.containers:
        if current_state.containers[container] == "closed":
            if allowed is None or "open" in allowed:
                actions.append(OpenAction(container=container))
        elif allowed is None or "close" in allowed:
            actions.append(CloseAction(container=container))

    for object_name, object_state in current_state.objects.items():
        current_holder = object_state.holder
        current_holder_closed = _is_closed_container(current_state, current_holder)

        if object_state.visibility == "visible":
            if allowed is None or "hide" in allowed:
                actions.append(HideAction(object=object_name))
        elif not current_holder_closed and (allowed is None or "reveal" in allowed):
            actions.append(RevealAction(object=object_name))

        if (allowed is None or "give" in allowed) and is_owner(current_state, current_holder):
            for owner in current_state.owners:
                if owner != current_holder:
                    actions.append(
                        GiveAction(
                            object=object_name,
                            from_owner=current_holder,
                            to_owner=owner,
                        )
                    )

        if current_holder_closed or (allowed is not None and "move" not in allowed):
            continue

        for target_holder in all_holders(current_state):
            if target_holder == current_holder:
                continue
            if is_owner(current_state, current_holder) and is_owner(current_state, target_holder):
                continue
            if _is_closed_container(current_state, target_holder):
                continue
            actions.append(
                MoveAction(object=object_name, from_=current_holder, to=target_holder)
            )

    return sorted(actions, key=action_key)


def action_key(action: ActionStruct) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return a deterministic key for an action."""

    parsed = parse_action_struct(action)
    data = parsed.model_dump(mode="json", by_alias=True)
    items = tuple(sorted((str(key), str(value)) for key, value in data.items()))
    return parsed.type, items


def actions_equal(left: ActionStruct, right: ActionStruct) -> bool:
    """Return whether two actions are identical by structure."""

    return action_key(left) == action_key(right)


def filter_distinct_actions(
    state: WorldState,
    positive_action: ActionStruct,
    candidates: Iterable[ActionStruct],
) -> list[ActionStruct]:
    """Keep only valid candidates with different resulting semantics."""

    positive_after = apply_action(state, positive_action)
    distinct: list[ActionStruct] = []
    for candidate in candidates:
        if actions_equal(candidate, positive_action):
            continue
        candidate_after = apply_action(state, candidate)
        if candidate_after.model_dump(mode="json") == positive_after.model_dump(mode="json"):
            continue
        distinct.append(candidate)
    return sorted(distinct, key=action_key)
