"""Message-shape parsing for the RQ3 sources.

These exist because of a failure that produced no error at all. The SWE-agent
trajectories tag the model's turn ``ai``, not ``assistant``; the loader matched on
``assistant``, yielded zero samples, and the sweep reported "0 groups have >= 38
samples" as though the dataset were unsuitable. Nothing crashed, nothing warned, and
the wrong conclusion was a plausible one.

Field and role vocabularies are per-dataset facts that cannot be guessed, so each one
the loaders rely on is pinned here. No network.
"""

import pytest

from specdecode.experiments.sources import (
    ALL_SOURCES,
    SPECS,
    Sample,
    _turns,
    group_histogram,
)


def test_wildchat_message_shape():
    conversation = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]
    assert list(_turns(conversation)) == [
        ("question one", "answer one"),
        ("question two", "answer two"),
    ]


def test_swetraj_message_shape():
    """role ``ai`` and field ``text``: the pair that silently produced nothing."""
    trajectory = [
        {"role": "system", "text": "you are an agent"},
        {"role": "user", "text": "the issue is X"},
        {"role": "ai", "text": "let me reproduce it"},
        {"role": "user", "text": "Traceback ..."},
        {"role": "ai", "text": "the error is a 403"},
    ]
    turns = list(_turns(trajectory, content_field="text"))
    assert turns == [
        ("the issue is X", "let me reproduce it"),
        ("Traceback ...", "the error is a 403"),
    ]


def test_unknown_model_role_yields_nothing_rather_than_wrong_pairs():
    """The failure mode is silence, so pin that it stays silence and not garbage."""
    conversation = [
        {"role": "user", "content": "q"},
        {"role": "chatbot", "content": "a"},
    ]
    assert list(_turns(conversation)) == []


def test_tool_turns_count_as_context():
    """In an agent loop the harness reply is the context for the next action."""
    conversation = [
        {"role": "tool", "content": "exit code 1"},
        {"role": "assistant", "content": "fixing the import"},
    ]
    assert list(_turns(conversation)) == [("exit code 1", "fixing the import")]


def test_a_model_turn_is_never_paired_with_a_later_input():
    """Two model turns in a row must not both consume the same preceding input."""
    conversation = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]
    assert list(_turns(conversation)) == [("q", "a1")]


def test_missing_content_is_tolerated():
    conversation = [
        {"role": "user"},
        {"role": "assistant", "content": None},
    ]
    assert list(_turns(conversation)) == [("", "")]


@pytest.mark.parametrize("name", ALL_SOURCES)
def test_every_source_declares_its_grouping(name):
    spec = SPECS[name]
    assert spec.grouping in ("natural", "none")
    # A source claiming a natural grouping must name the field it comes from,
    # otherwise "natural" is an assertion nobody can check against the dataset card.
    assert bool(spec.group_field) == (spec.grouping == "natural")


def test_group_histogram_counts_samples_per_group():
    samples = [
        Sample(group_id="a", doc_text="p", target_text="t"),
        Sample(group_id="a", doc_text="p", target_text="t"),
        Sample(group_id="b", doc_text="p", target_text="t"),
    ]
    assert group_histogram(samples) == {"a": 2, "b": 1}
