"""History scoping as a data transformation.

RQ3 asks whether restricting *which history* enters the datastore raises workload
speculatability without touching the drafting algorithm. So a scope must be a pure
data object: it selects documents, and the existing ``IndexedTensorNGramDrafter``
runs on whatever it selects. Nothing here knows how drafting works.

Leakage is prevented structurally rather than by review. Every policy receives the
history through :func:`visible_history`, which drops anything not strictly earlier
than the request being served; a policy cannot widen that set because it never sees
the unfiltered one. The request's own source document is separate -- it is the input
to the task (the article being summarised), not the answer, exactly as in the RQ2
``self_relevant`` construction.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Document:
    """One piece of servable history: a source document that was seen earlier."""

    doc_id: str
    group_id: str  # the real grouping key: user, repo, article title, database, ...
    order: int  # position in the global serving sequence
    tokens: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class Request:
    """One evaluated request: generate ``target_tokens`` given a source document."""

    request_id: str
    group_id: str
    order: int
    doc_tokens: Tuple[int, ...]  # the task input, always available
    target_tokens: Tuple[int, ...]  # what the target model must produce


def visible_history(history: Sequence[Document], request: Request) -> List[Document]:
    """Documents a request is allowed to see: strictly earlier in serving order.

    This is the single choke point for causality. Policies call it and cannot
    bypass it, so "the datastore contained a future request's material" is not a
    mistake this module can make.
    """
    return [doc for doc in history if doc.order < request.order]


@dataclass(frozen=True)
class Scope:
    """A resolved datastore: the tokens plus the key that identifies its build.

    ``key`` is what the index cache is keyed on, so it must determine the tokens
    exactly. Requests sharing a key share one built index, and the number of them
    *is* the amortization factor ``R`` in the RQ2 cost model -- counted, not assumed.
    """

    key: str
    tokens: List[int]
    n_documents: int


class ScopePolicy:
    """Base class: pick history documents, then flatten them into a datastore.

    A scope contains **history only**. The request's own source document is part of
    its prompt, not of the retained history, and mixing the two would make the key
    a lie: two requests selecting the same history would still need different
    datastores, so an index keyed on the history would serve one of them the other's
    document. ``RequestLocal`` is the deliberate exception -- there the prompt *is*
    the datastore, which is what Fast Context means.
    """

    name = "base"

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        raise NotImplementedError

    def resolve(self, request: Request, history: Sequence[Document]) -> Scope:
        key, docs = self.select(request, visible_history(history, request))
        tokens: List[int] = []
        for doc in docs:
            tokens.extend(doc.tokens)
        return Scope(key=key, tokens=tokens, n_documents=len(docs))


class RequestLocal(ScopePolicy):
    """Fast Context: the request's own source document and nothing else.

    Maximum relevance, minimum support, and no reuse at all -- every request pays a
    fresh index build, which is why this arm is expected to lose ground once build
    cost is included.
    """

    name = "request_local"

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        """No history at all -- the datastore is built from the prompt in ``resolve``.

        This returns an empty selection rather than ``NotImplementedError`` so that
        :func:`assert_causal` can still run over this policy. The causality claim it
        checks here is the real one: request-local retains **nothing** from earlier
        requests, so there is nothing that could have leaked.
        """
        return f"local|{request.request_id}", []

    def resolve(self, request: Request, history: Sequence[Document]) -> Scope:
        return Scope(
            key=f"local|{request.request_id}",
            tokens=list(request.doc_tokens),
            n_documents=1,
        )


class GroupScoped(ScopePolicy):
    """DAS: history from the same group as the request, and nothing else.

    The group is a **real** key carried by the data itself -- the user who sent the
    request, the repository a function came from, the Wikipedia article a question
    was written about, the database a query targets. It is never the dataset name:
    the pilot used that and group scope came out indistinguishable from global
    (0.1367 vs 0.1366), because text from an unrelated dataset is simply inert.
    """

    name = "group"

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        docs = [d for d in visible if d.group_id == request.group_id]
        return f"group|{request.group_id}", docs


class NarrowScoped(ScopePolicy):
    """Negative control: the ``n_docs`` most recent documents of the same group.

    This cuts on **recency**, which is not correlated with relevance, so it is the
    control that shows narrowing per se buys nothing: the pilot's oracle rate fell
    monotonically along this ladder (0.0948 -> 0.0561 -> 0.0260) while request-local,
    at a tenth the size, scored 0.0964. Sweeping ``n_docs`` also turns three points
    into a curve, which is what lets the sweet spot be located rather than asserted.
    """

    def __init__(self, n_docs: int) -> None:
        if n_docs < 0:
            raise ValueError("n_docs must be non-negative")
        self.n_docs = n_docs
        self.name = f"narrow_{n_docs}"

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        docs = [d for d in visible if d.group_id == request.group_id]
        docs = docs[-self.n_docs :] if self.n_docs else []
        # The window slides with serving order, so requests share a key only while
        # the window is unchanged. Keying on the boundary keeps reuse honest.
        boundary = docs[0].order if docs else -1
        return f"narrow{self.n_docs}|{request.group_id}|{boundary}", docs


class GlobalScope(ScopePolicy):
    """REST: every document served so far, from every group."""

    name = "global"

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        return "global", list(visible)


@dataclass
class SizeMatchedControl(ScopePolicy):
    """A treatment's twin: same datastore size, same reuse structure, random content.

    Narrowing a scope shrinks the corpus and concentrates relevance at the same time,
    and RQ2 showed corpus size moves both ``L/S`` and ``t_d`` on its own. The gap
    between a treatment and this control is the part attributable to relevance.

    It mirrors the treatment rather than merely matching a mean, on two axes:

    * **size, request by request** -- groups differ by an order of magnitude in
      document length, so one pooled target would hand the short-document groups a
      datastore several times larger than the treatment it controls for;
    * **bucketing** -- the control inherits the treatment's scope key, so it builds
      the same number of indexes and carries the same reuse factor ``R``. Drawing a
      fresh corpus per request instead would make the control pay 200 index builds
      against the treatment's 5, and the including-build comparison would measure
      that artifact rather than relevance.
    """

    treatment: ScopePolicy
    seed: int = 0
    name: str = field(default="control", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"control_{self.treatment.name}")

    def select(
        self, request: Request, visible: Sequence[Document]
    ) -> Tuple[str, List[Document]]:
        target = self.treatment.resolve(request, list(visible))
        budget = len(target.tokens)
        key = f"control|{target.key}"
        if not visible or budget == 0:
            return key, []
        # Seeded from the key, not the request, so every request sharing a key draws
        # the same corpus -- otherwise the key would not determine the tokens.
        rng = np.random.default_rng(
            [self.seed, budget, int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)]
        )
        order = rng.permutation(len(visible))
        chosen: List[Document] = []
        total = 0
        for idx in order:
            if total >= budget:
                break
            doc = visible[int(idx)]
            chosen.append(doc)
            total += len(doc)
        return key, chosen

    def resolve(self, request: Request, history: Sequence[Document]) -> Scope:
        """Trim to the treatment's size exactly.

        Whole documents overshoot badly at small budgets -- a 146-token prompt
        cannot be matched with 1024-token articles -- and an oversized control is
        no longer a control for size.
        """
        budget = len(self.treatment.resolve(request, history).tokens)
        scope = super().resolve(request, history)
        return Scope(
            key=scope.key,
            tokens=scope.tokens[:budget],
            n_documents=scope.n_documents,
        )


def group_sizes(documents: Sequence[Document]) -> Dict[str, int]:
    """Documents per group -- reported so a "group" of one is never mistaken for a result."""
    sizes: Dict[str, int] = {}
    for doc in documents:
        sizes[doc.group_id] = sizes.get(doc.group_id, 0) + 1
    return sizes


def scope_sizes(
    policy: ScopePolicy, requests: Sequence[Request], history: Sequence[Document]
) -> List[int]:
    """Datastore token count per request -- used to size the matched control."""
    return [len(policy.resolve(r, history).tokens) for r in requests]


def fingerprint(tokens: Sequence[int]) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    """Cheap identity for a datastore: length plus both ends."""
    return (len(tokens), tuple(tokens[:32]), tuple(tokens[-32:]))


def assert_key_determines_tokens(
    policy: ScopePolicy, requests: Sequence[Request], history: Sequence[Document]
) -> int:
    """Two requests sharing a scope key must resolve to the same datastore.

    An index is cached per key, so if a key can map to two different token lists one
    of those requests silently gets the other's datastore -- which is a measurement
    error that looks like a result. This is asserted rather than assumed.
    """
    seen: Dict[str, Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = {}
    for request in requests:
        scope = policy.resolve(request, history)
        mark = fingerprint(scope.tokens)
        if scope.key in seen and seen[scope.key] != mark:
            raise AssertionError(
                f"scope key {scope.key!r} maps to two different datastores "
                f"(request {request.request_id}); the key does not determine the tokens"
            )
        seen[scope.key] = mark
    return len(seen)


def assert_causal(
    policy: ScopePolicy,
    requests: Sequence[Request],
    history: Sequence[Document],
    *,
    limit: Optional[int] = None,
) -> int:
    """Re-derive the selection and fail loudly if anything non-earlier slipped in.

    :func:`visible_history` already makes this impossible by construction; this is
    the assertion that keeps it that way after future edits, and the runner calls it
    before recording any measurement.
    """
    checked = 0
    subset = requests if limit is None else requests[:limit]
    for request in subset:
        _, docs = policy.select(request, visible_history(history, request))
        for doc in docs:
            if doc.order >= request.order:
                raise AssertionError(
                    f"leakage: {policy.name} gave request {request.request_id} "
                    f"(order {request.order}) document {doc.doc_id} at order {doc.order}"
                )
        checked += 1
    return checked
