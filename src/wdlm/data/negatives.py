"""Near-miss negative update generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from wdlm.data.actions import action_key, filter_distinct_actions, list_valid_actions
from wdlm.data.renderer import render_action, template_id_for_action
from wdlm.schemas import (
    ActionStruct,
    CloseAction,
    GiveAction,
    HideAction,
    MoveAction,
    NegativeType,
    NegativeUpdate,
    OpenAction,
    RevealAction,
    WorldState,
    parse_action_struct,
)


Predicate = Callable[[ActionStruct], bool]


def _wrong_object(positive: ActionStruct) -> Predicate:
    positive = parse_action_struct(positive)

    def predicate(candidate: ActionStruct) -> bool:
        candidate = parse_action_struct(candidate)
        if positive.type != candidate.type:
            return False
        if not hasattr(positive, "object") or not hasattr(candidate, "object"):
            return False
        if positive.object == candidate.object:
            return False
        if isinstance(positive, MoveAction) and isinstance(candidate, MoveAction):
            return positive.from_ == candidate.from_ and positive.to == candidate.to
        if isinstance(positive, GiveAction) and isinstance(candidate, GiveAction):
            return (
                positive.from_owner == candidate.from_owner
                and positive.to_owner == candidate.to_owner
            )
        return True

    return predicate


def _wrong_destination(positive: ActionStruct) -> Predicate:
    positive = parse_action_struct(positive)

    def predicate(candidate: ActionStruct) -> bool:
        candidate = parse_action_struct(candidate)
        if isinstance(positive, MoveAction) and isinstance(candidate, MoveAction):
            return positive.object == candidate.object and positive.to != candidate.to
        if isinstance(positive, GiveAction) and isinstance(candidate, GiveAction):
            return positive.object == candidate.object and positive.to_owner != candidate.to_owner
        if isinstance(positive, (OpenAction, CloseAction)) and isinstance(
            candidate, (OpenAction, CloseAction)
        ):
            return positive.container != candidate.container
        return False

    return predicate


def _opposite_operation(positive: ActionStruct) -> Predicate:
    positive = parse_action_struct(positive)

    def predicate(candidate: ActionStruct) -> bool:
        candidate = parse_action_struct(candidate)
        if isinstance(positive, OpenAction) and isinstance(candidate, CloseAction):
            return positive.container == candidate.container
        if isinstance(positive, CloseAction) and isinstance(candidate, OpenAction):
            return positive.container == candidate.container
        if isinstance(positive, HideAction) and isinstance(candidate, RevealAction):
            return positive.object == candidate.object
        if isinstance(positive, RevealAction) and isinstance(candidate, HideAction):
            return positive.object == candidate.object
        return False

    return predicate


def _wrong_owner(positive: ActionStruct) -> Predicate:
    positive = parse_action_struct(positive)

    def predicate(candidate: ActionStruct) -> bool:
        candidate = parse_action_struct(candidate)
        if not isinstance(positive, GiveAction) or not isinstance(candidate, GiveAction):
            return False
        if positive.object != candidate.object:
            return False
        return (
            positive.from_owner != candidate.from_owner
            or positive.to_owner != candidate.to_owner
        )

    return predicate


NEGATIVE_MATCHERS: tuple[tuple[NegativeType, Callable[[ActionStruct], Predicate]], ...] = (
    ("wrong_object", _wrong_object),
    ("wrong_destination", _wrong_destination),
    ("opposite_operation", _opposite_operation),
    ("wrong_owner", _wrong_owner),
)


def generate_negative_updates(
    state: WorldState,
    positive_action: ActionStruct,
    *,
    max_negatives: int = 4,
    allowed_action_types: set[str] | None = None,
    candidate_actions: Iterable[ActionStruct] | None = None,
) -> list[NegativeUpdate]:
    """Generate deterministic near-miss negative actions for a positive update."""

    positive_action = parse_action_struct(positive_action)
    candidates_source = (
        list(candidate_actions)
        if candidate_actions is not None
        else list_valid_actions(state, allowed_action_types=allowed_action_types)
    )
    candidates = filter_distinct_actions(
        state=state,
        positive_action=positive_action,
        candidates=candidates_source,
    )

    selected: list[tuple[ActionStruct, NegativeType]] = []
    used_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for negative_type, matcher_factory in NEGATIVE_MATCHERS:
        matcher = matcher_factory(positive_action)
        for candidate in candidates:
            candidate_key = action_key(candidate)
            if candidate_key in used_keys:
                continue
            if matcher(candidate):
                selected.append((candidate, negative_type))
                used_keys.add(candidate_key)
                break

    for candidate in candidates:
        candidate_key = action_key(candidate)
        if candidate_key in used_keys:
            continue
        selected.append((candidate, "fallback"))
        used_keys.add(candidate_key)
        if len(selected) >= max_negatives:
            break

    updates: list[NegativeUpdate] = []
    for action, negative_type in selected[:max_negatives]:
        template_index = 0
        updates.append(
            NegativeUpdate(
                action_struct=action,
                text_chunk=render_action(action, template_index=template_index),
                negative_type=negative_type,
                template_id=template_id_for_action(action, template_index),
            )
        )
    return updates
