from __future__ import annotations

from wdlm.data.generate import generate_toy_world_examples
from wdlm.data.queries import generate_query_records


def test_generated_queries_match_state_answers() -> None:
    example = generate_toy_world_examples(num_examples=1, seed=13)[0]
    queries = generate_query_records([example], seed=13)
    answers_by_type = {query.question_type: query.answer for query in queries}

    where_object_name = sorted(example.state_after.objects)[0]
    owner_object_name = sorted(example.state_after.objects)[1 % len(example.state_after.objects)]
    visibility_object_name = sorted(example.state_after.objects)[2 % len(example.state_after.objects)]
    container_name = sorted(example.state_after.containers)[0]

    assert answers_by_type["where_is_object"] == example.state_after.objects[where_object_name].holder
    expected_owner = example.state_after.objects[owner_object_name].holder
    if expected_owner not in example.state_after.owners:
        expected_owner = "nobody"
    assert answers_by_type["who_owns_object"] == expected_owner
    assert answers_by_type["is_container_open"] in {"yes", "no"}
    expected_visibility = (
        "yes"
        if example.state_after.objects[visibility_object_name].visibility == "visible"
        else "no"
    )
    assert answers_by_type["is_object_visible"] == expected_visibility
    assert all(query.example_id == example.example_id for query in queries)
