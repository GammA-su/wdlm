"""Deterministic surface rendering for toy-world actions."""

from __future__ import annotations

from collections.abc import Callable

from wdlm.data.toy_world import OWNER_POOL
from wdlm.schemas import ActionStruct, parse_action_struct


TemplateFn = Callable[[ActionStruct], str]


def humanize_name(name: str) -> str:
    """Convert an identifier like ``red_key`` into ``red key``."""

    return name.replace("_", " ")


def holder_phrase(name: str) -> str:
    """Render a holder name with an article when appropriate."""

    text = humanize_name(name)
    if name in OWNER_POOL:
        return text
    return f"the {text}"


def _move_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: (
            f"Move the {humanize_name(action.object)} from {holder_phrase(action.from_)} "
            f"to {holder_phrase(action.to)}."
        ),
        lambda action: (
            f"Place the {humanize_name(action.object)} at {holder_phrase(action.to)} "
            f"instead of {holder_phrase(action.from_)}."
        ),
        lambda action: (
            f"Shift the {humanize_name(action.object)} out of {holder_phrase(action.from_)} "
            f"and into {holder_phrase(action.to)}."
        ),
        lambda action: (
            f"The {humanize_name(action.object)} goes from {holder_phrase(action.from_)} "
            f"to {holder_phrase(action.to)}."
        ),
    )


def _open_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: f"Open the {humanize_name(action.container)}.",
        lambda action: f"Pull the {humanize_name(action.container)} open.",
        lambda action: f"Set the {humanize_name(action.container)} to the open state.",
        lambda action: f"Make sure the {humanize_name(action.container)} is open.",
    )


def _close_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: f"Close the {humanize_name(action.container)}.",
        lambda action: f"Shut the {humanize_name(action.container)}.",
        lambda action: f"Set the {humanize_name(action.container)} to the closed state.",
        lambda action: f"Make sure the {humanize_name(action.container)} is closed.",
    )


def _give_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: (
            f"Give the {humanize_name(action.object)} from {humanize_name(action.from_owner)} "
            f"to {humanize_name(action.to_owner)}."
        ),
        lambda action: (
            f"Hand the {humanize_name(action.object)} to {humanize_name(action.to_owner)} "
            f"from {humanize_name(action.from_owner)}."
        ),
        lambda action: (
            f"Transfer the {humanize_name(action.object)} from "
            f"{humanize_name(action.from_owner)} over to {humanize_name(action.to_owner)}."
        ),
        lambda action: (
            f"The {humanize_name(action.object)} should pass from "
            f"{humanize_name(action.from_owner)} to {humanize_name(action.to_owner)}."
        ),
    )


def _hide_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: f"Hide the {humanize_name(action.object)}.",
        lambda action: f"Make the {humanize_name(action.object)} hidden.",
        lambda action: f"Keep the {humanize_name(action.object)} out of sight.",
        lambda action: f"The {humanize_name(action.object)} should no longer be visible.",
    )


def _reveal_templates() -> tuple[TemplateFn, ...]:
    return (
        lambda action: f"Reveal the {humanize_name(action.object)}.",
        lambda action: f"Make the {humanize_name(action.object)} visible.",
        lambda action: f"Bring the {humanize_name(action.object)} into view.",
        lambda action: f"The {humanize_name(action.object)} should now be visible.",
    )


TEMPLATES: dict[str, tuple[TemplateFn, ...]] = {
    "move": _move_templates(),
    "open": _open_templates(),
    "close": _close_templates(),
    "give": _give_templates(),
    "hide": _hide_templates(),
    "reveal": _reveal_templates(),
}


def template_count(action: ActionStruct) -> int:
    """Return the number of templates for an action type."""

    parsed = parse_action_struct(action)
    return len(TEMPLATES[parsed.type])


def template_id_for_action(action: ActionStruct, template_index: int) -> str:
    """Return a stable template identifier for an action/template pair."""

    parsed = parse_action_struct(action)
    return f"{parsed.type}_t{template_index}"


def render_action(action: ActionStruct, template_index: int = 0) -> str:
    """Render an action using one deterministic template variant."""

    parsed = parse_action_struct(action)
    templates = TEMPLATES[parsed.type]
    if template_index < 0 or template_index >= len(templates):
        raise ValueError(
            f"Template index {template_index} out of range for action type {parsed.type}."
        )
    return templates[template_index](parsed)


def render_template_entries(
    action: ActionStruct,
    *,
    exclude_template: int | None = None,
) -> list[tuple[str, str]]:
    """Return ``(template_id, text)`` pairs for every template variant."""

    entries: list[tuple[str, str]] = []
    for template_index in range(template_count(action)):
        if exclude_template is not None and template_index == exclude_template:
            continue
        entries.append(
            (
                template_id_for_action(action, template_index),
                render_action(action, template_index=template_index),
            )
        )
    return entries
