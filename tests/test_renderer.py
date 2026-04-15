from __future__ import annotations

from wdlm.data.paraphrase import generate_paraphrases
from wdlm.data.renderer import render_action
from wdlm.schemas import MoveAction, OpenAction


def test_move_renderer_default_template() -> None:
    action = MoveAction(object="red_key", from_="drawer", to="table")
    assert render_action(action) == "Move the red key from the drawer to the table."


def test_paraphrases_are_deterministic_and_distinct() -> None:
    action = OpenAction(container="drawer")
    paraphrases = generate_paraphrases(action, exclude_template=0)
    assert paraphrases == [
        "Pull the drawer open.",
        "Set the drawer to the open state.",
        "Make sure the drawer is open.",
    ]
