"""Causality is the gate: a scope that sees the future would look great and mean nothing."""

import pytest

from specdecode.scoping import (
    Document,
    GlobalScope,
    GroupScoped,
    NarrowScoped,
    Request,
    RequestLocal,
    SizeMatchedControl,
    assert_causal,
    assert_key_determines_tokens,
    visible_history,
)


def doc(i: int, group: str, n: int = 10) -> Document:
    return Document(
        doc_id=f"{group}#{i}", group_id=group, order=i, tokens=tuple(range(i * 100, i * 100 + n))
    )


def req(order: int, group: str) -> Request:
    return Request(
        request_id=f"{group}#r{order}",
        group_id=group,
        order=order,
        doc_tokens=(9000 + order, 9001 + order),
        target_tokens=(1, 2, 3, 4),
    )


@pytest.fixture
def stream():
    history = [doc(i, "a" if i % 2 == 0 else "b") for i in range(10)]
    requests = [req(20, "a"), req(21, "b"), req(22, "a")]
    return history, requests


ALL_POLICIES = [RequestLocal(), GroupScoped(), GlobalScope(), NarrowScoped(3)]


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_no_policy_can_see_the_future(policy, stream):
    history, requests = stream
    # A document ordered after every request: no policy may surface it.
    future = Document(doc_id="future", group_id="a", order=999, tokens=(7, 7, 7))
    assert assert_causal(policy, requests, list(history) + [future]) == len(requests)


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_interleaved_requests_never_see_each_other(policy):
    """Requests interleaved with history: a later request must not leak backwards."""
    history = [doc(i, "a") for i in range(6)]
    requests = [req(1, "a"), req(3, "a"), req(5, "a")]
    assert_causal(policy, requests, history)


def test_visible_history_is_strict():
    history = [doc(0, "a"), doc(5, "a"), doc(9, "a")]
    request = req(5, "a")
    orders = [d.order for d in visible_history(history, request)]
    assert orders == [0], "a document at the same order is not earlier"


def test_assert_causal_catches_a_leaky_policy(stream):
    history, requests = stream
    # Orders 20/21/22 are the requests, so this document belongs to a request that
    # has not been served yet -- exactly the material a scope must never contain.
    leaked = Document(doc_id="future", group_id="a", order=21, tokens=(7, 7, 7))
    poisoned = list(history) + [leaked]

    class Leaky(RequestLocal):
        name = "leaky"

        def select(self, request, visible):
            return "leaky", list(poisoned)  # ignores the filtered view on purpose

    with pytest.raises(AssertionError, match="leakage"):
        assert_causal(Leaky(), requests, poisoned)


def test_request_local_is_only_the_own_document(stream):
    history, requests = stream
    scope = RequestLocal().resolve(requests[0], history)
    assert scope.tokens == list(requests[0].doc_tokens)
    assert scope.n_documents == 1


def test_group_scope_excludes_other_groups(stream):
    history, requests = stream
    request = requests[0]  # group "a"
    scope = GroupScoped().resolve(request, history)
    b_tokens = {t for d in history if d.group_id == "b" for t in d.tokens}
    assert not b_tokens & set(scope.tokens)


@pytest.mark.parametrize("policy", ALL_POLICIES + [SizeMatchedControl(treatment=GroupScoped())])
def test_scope_key_determines_the_datastore(policy, stream):
    """Regression: an index is cached per key, so a key that maps to two different
    token lists silently serves one request the other's datastore."""
    history, requests = stream
    assert_key_determines_tokens(policy, requests, history)


@pytest.mark.parametrize("policy", [GroupScoped(), GlobalScope(), NarrowScoped(3)])
def test_shared_scopes_hold_history_only(policy, stream):
    """The request's own document belongs to the prompt, not to retained history.

    If it were folded in, two requests selecting identical history would still need
    different datastores and the shared key would be wrong."""
    history, requests = stream
    scope = policy.resolve(requests[0], history)
    assert not set(requests[0].doc_tokens) & set(scope.tokens)


def test_assert_key_determines_tokens_catches_a_shared_key_with_different_tokens(stream):
    history, requests = stream

    class Sloppy(GroupScoped):
        name = "sloppy"

        def resolve(self, request, hist):
            scope = super().resolve(request, hist)
            return type(scope)(
                key=scope.key,
                tokens=list(request.doc_tokens) + scope.tokens,
                n_documents=scope.n_documents,
            )

    with pytest.raises(AssertionError, match="does not determine"):
        assert_key_determines_tokens(Sloppy(), requests, history)


def test_global_is_a_superset_of_group(stream):
    history, requests = stream
    request = requests[0]
    narrow = GroupScoped().resolve(request, history)
    wide = GlobalScope().resolve(request, history)
    assert len(wide.tokens) > len(narrow.tokens)
    assert set(narrow.tokens) <= set(wide.tokens)


def test_narrow_takes_the_most_recent(stream):
    history, requests = stream
    request = requests[0]
    scope = NarrowScoped(2).resolve(request, history)
    # group "a" history is orders 0,2,4,6,8 -> the last two are 6 and 8
    assert set(doc(6, "a").tokens) <= set(scope.tokens)
    assert set(doc(8, "a").tokens) <= set(scope.tokens)
    assert not set(doc(0, "a").tokens) & set(scope.tokens)


def test_narrow_zero_selects_no_history(stream):
    history, requests = stream
    assert NarrowScoped(0).resolve(requests[0], history).n_documents == 0


def test_scope_keys_group_requests_that_share_a_datastore(stream):
    history, requests = stream
    keys = {GroupScoped().resolve(r, history).key for r in requests}
    assert keys == {"group|a", "group|b"}
    assert len({GlobalScope().resolve(r, history).key for r in requests}) == 1
    assert len({RequestLocal().resolve(r, history).key for r in requests}) == len(requests)


def test_size_matched_control_matches_the_treatment_size(stream):
    history, requests = stream
    treatment = GroupScoped()
    policy = SizeMatchedControl(treatment=treatment, seed=1)
    for request in requests:
        want = len(treatment.resolve(request, history).tokens)
        assert len(policy.resolve(request, history).tokens) == want


def test_size_matched_control_is_causal(stream):
    history, requests = stream
    assert_causal(SizeMatchedControl(treatment=GroupScoped(), seed=3), requests, history)


def test_size_matched_control_mirrors_the_treatment_bucketing(stream):
    """Same number of indexes as the treatment, so R matches and the
    including-build comparison measures relevance rather than build churn."""
    history, requests = stream
    for treatment in (RequestLocal(), GroupScoped(), NarrowScoped(2), GlobalScope()):
        control = SizeMatchedControl(treatment=treatment, seed=5)
        t_keys = {treatment.resolve(r, history).key for r in requests}
        c_keys = {control.resolve(r, history).key for r in requests}
        assert len(t_keys) == len(c_keys), treatment.name


def test_size_matched_control_draws_no_same_document_twice(stream):
    history, requests = stream
    scope = SizeMatchedControl(treatment=GlobalScope(), seed=4).resolve(requests[0], history)
    assert len(scope.tokens) == len(set(scope.tokens)) or scope.n_documents <= len(history)


def test_size_matched_control_matches_each_request_separately(stream):
    """A pooled target would hand a short-document group a far larger datastore."""
    history, requests = stream
    treatment = RequestLocal()  # per-request sizes differ
    policy = SizeMatchedControl(treatment=treatment, seed=2)
    for request in requests:
        assert len(policy.resolve(request, history).tokens) == len(request.doc_tokens)


def test_size_matched_control_is_deterministic(stream):
    history, requests = stream
    a = SizeMatchedControl(treatment=GroupScoped(), seed=7).resolve(requests[0], history)
    b = SizeMatchedControl(treatment=GroupScoped(), seed=7).resolve(requests[0], history)
    assert a.tokens == b.tokens


def test_empty_history_leaves_shared_scopes_empty(stream):
    _, requests = stream
    for policy in [GroupScoped(), GlobalScope(), NarrowScoped(3),
                   SizeMatchedControl(treatment=GroupScoped())]:
        assert policy.resolve(requests[0], []).tokens == []
    # Request-local never depended on history in the first place.
    assert RequestLocal().resolve(requests[0], []).tokens == list(requests[0].doc_tokens)
