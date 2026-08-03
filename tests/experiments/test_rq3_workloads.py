"""The RQ3 data layer, which is where the pilot actually went wrong.

The scoping machinery was already correct and tested; what made the pilot
uninterpretable was that its group key was the dataset name. These tests pin the
properties that make a group real: several requests share it, history is what an
earlier request produced, and a source without a key is refused rather than quietly
turned into a ladder with a duplicated rung.

No network. Samples are synthesised so the invariants are checked, not the mood of
a Hugging Face mirror.
"""

import pytest

from specdecode.experiments.rq3_workloads import (
    NotEnoughGroups,
    build_stream,
    select_groups,
    stream_key,
)
from specdecode.experiments.sources import Sample, group_histogram
from specdecode.scoping import (
    GlobalScope,
    GroupScoped,
    RequestLocal,
    SizeMatchedControl,
    assert_causal,
    assert_key_determines_tokens,
    group_sizes,
)


def enc(text: str):
    """Deterministic stand-in tokenizer: one token per character, stable per string."""
    return [ord(c) % 500 for c in text]


def samples(n_groups: int, per_group: int) -> list:
    out = []
    for g in range(n_groups):
        for i in range(per_group):
            out.append(
                Sample(
                    group_id=f"g{g}",
                    doc_text=f"prompt for group {g} item {i} " * 3,
                    target_text=f"answer of group {g} item {i} " * 3,
                )
            )
    return out


@pytest.fixture
def stream():
    return build_stream(
        samples(4, 8), enc,
        history_per_group=5, requests_per_group=3,
        doc_token_cap=1024, target_token_cap=512, max_groups=10,
    )


def test_groups_are_shared_by_several_requests(stream):
    """The whole point: a group with one request is request-local under another name."""
    _, requests = stream
    counts = {}
    for request in requests:
        counts[request.group_id] = counts.get(request.group_id, 0) + 1
    assert len(counts) == 4
    assert all(n > 1 for n in counts.values())


def test_history_carries_the_generated_answer_not_just_the_prompt(stream):
    """A group scope of prompts with no answers has nothing a drafter can copy."""
    documents, _ = stream
    doc = documents[0]
    prompt_only = enc("prompt for group 0 item 0 " * 3)
    assert len(doc.tokens) > len(prompt_only)
    assert list(doc.tokens[: len(prompt_only)]) == prompt_only


def test_request_local_datastore_is_the_prompt_and_never_the_answer(stream):
    documents, requests = stream
    request = requests[0]
    scope = RequestLocal().resolve(request, documents)
    assert scope.tokens == list(request.doc_tokens)
    assert list(request.target_tokens) != scope.tokens


def test_history_precedes_every_request(stream):
    documents, requests = stream
    assert max(d.order for d in documents) < min(r.order for r in requests)


def test_groups_are_interleaved_not_contiguous(stream):
    """Contiguous blocks would make narrow(N) return the group scope for early requests."""
    documents, _ = stream
    first_two = [d.group_id for d in documents[:2]]
    assert first_two[0] != first_two[1]


@pytest.mark.parametrize("policy", [GlobalScope(), GroupScoped(), RequestLocal()])
def test_every_policy_is_causal_and_key_determined(stream, policy):
    documents, requests = stream
    assert assert_causal(policy, requests, documents) == len(requests)
    assert assert_key_determines_tokens(policy, requests, documents) >= 1


def test_group_scope_is_smaller_than_global_but_not_empty(stream):
    documents, requests = stream
    request = requests[0]
    group = GroupScoped().resolve(request, documents)
    world = GlobalScope().resolve(request, documents)
    assert 0 < len(group.tokens) < len(world.tokens)


def test_group_scope_contains_only_its_own_group(stream):
    documents, requests = stream
    request = requests[0]
    _, docs = GroupScoped().select(request, documents)
    assert {d.group_id for d in docs} == {request.group_id}


def test_control_matches_treatment_size(stream):
    """A control that is not the treatment's size is not controlling for size."""
    documents, requests = stream
    treatment = GroupScoped()
    control = SizeMatchedControl(treatment=treatment, seed=1)
    for request in requests:
        want = len(treatment.resolve(request, documents).tokens)
        got = len(control.resolve(request, documents).tokens)
        assert got == want


def test_control_content_differs_from_treatment(stream):
    """Same size, different tokens -- otherwise the control measures nothing."""
    documents, requests = stream
    treatment = GroupScoped()
    control = SizeMatchedControl(treatment=treatment, seed=1)
    request = requests[0]
    assert control.resolve(request, documents).tokens != treatment.resolve(request, documents).tokens


def test_singleton_groups_are_refused():
    """Sources with no grouping must fail loudly rather than produce a fake ladder."""
    one_each = [Sample(group_id=f"g{i}", doc_text="a b c", target_text="d e f") for i in range(9)]
    with pytest.raises(NotEnoughGroups):
        build_stream(
            one_each, enc,
            history_per_group=5, requests_per_group=3,
            doc_token_cap=64, target_token_cap=64, max_groups=10,
        )


def test_singleton_groups_allowed_when_declared():
    """SAMSum has no key and is kept deliberately, so the escape hatch must work."""
    one_each = [
        Sample(group_id=f"g{i}", doc_text="a b c d e", target_text="f g h i j")
        for i in range(9)
    ]
    documents, requests = build_stream(
        one_each, enc,
        history_per_group=1, requests_per_group=1,
        doc_token_cap=64, target_token_cap=64, max_groups=10,
        allow_singleton_groups=True,
    )
    assert documents and requests


def test_group_histogram_and_selection_agree():
    data = samples(3, 6)
    counts = group_histogram(data)
    assert set(counts) == {"g0", "g1", "g2"}
    assert select_groups(data, min_per_group=6, max_groups=2) == ["g0", "g1"]
    assert select_groups(data, min_per_group=7, max_groups=5) == []


def test_group_sizes_counts_documents(stream):
    documents, _ = stream
    assert sum(group_sizes(documents).values()) == len(documents)


def test_stream_key_is_stable_and_distinguishing():
    a = stream_key("squad", 40, 10, 1024, 512, 20)
    b = stream_key("squad", 40, 10, 1024, 512, 21)
    assert a == stream_key("squad", 40, 10, 1024, 512, 20)
    assert a != b


def test_targets_are_never_padded_to_a_fixed_length():
    """The pilot concatenated summaries to reach 200 tokens; that request is fictional."""
    mixed = [
        Sample(group_id="g", doc_text="p" * 40, target_text="short"),
        Sample(group_id="g", doc_text="p" * 40, target_text="a much longer answer here"),
        Sample(group_id="g", doc_text="p" * 40, target_text="medium answer"),
        Sample(group_id="g2", doc_text="p" * 40, target_text="another"),
        Sample(group_id="g2", doc_text="p" * 40, target_text="yet another answer"),
        Sample(group_id="g2", doc_text="p" * 40, target_text="third one"),
    ]
    _, requests = build_stream(
        mixed, enc,
        history_per_group=1, requests_per_group=2,
        doc_token_cap=64, target_token_cap=64, max_groups=4,
    )
    lengths = {len(r.target_tokens) for r in requests}
    assert len(lengths) > 1
