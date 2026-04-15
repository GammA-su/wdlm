"""Templatic paraphrase generation."""

from __future__ import annotations

from wdlm.data.renderer import render_template_entries
from wdlm.schemas import ActionStruct


def generate_paraphrase_entries(
    action: ActionStruct,
    *,
    exclude_template: int | None = None,
) -> list[tuple[str, str]]:
    """Return deterministic paraphrase ``(template_id, text)`` pairs for an action."""

    return render_template_entries(action, exclude_template=exclude_template)


def generate_paraphrases(
    action: ActionStruct,
    *,
    exclude_template: int | None = None,
) -> list[str]:
    """Return deterministic paraphrases for an action."""

    return [
        text
        for _, text in generate_paraphrase_entries(
            action,
            exclude_template=exclude_template,
        )
    ]
