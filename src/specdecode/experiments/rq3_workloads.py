"""Turn a group-aware source into a serving stream of history and requests.

The whole experiment turns on one modelling choice: **a history document is an
earlier request's prompt together with what was generated for it.** That is what a
real system retains, and it is what makes a group scope worth anything -- the user's
previous answers, the repository's previous functions, the schema's previous
queries. The pilot retained source documents only and grouped them by dataset name,
which is why its group scope was indistinguishable from global.

    history for group G  =  concat(prompt, response) of G's earlier samples
    request in group G   =  prompt (local scope)  ->  response (the target)

History is laid down before any request is served, and stays fixed. That keeps a
scope key stable, so an index is built once and the reuse factor ``R`` measured by
``ScopedIndexCache`` describes the scope rather than a cache-invalidation policy. It
also means causality holds trivially, which :func:`assert_causal` reports honestly
rather than dressing up as a finding.

Targets are used at their natural length, capped but never padded. The pilot
concatenated consecutive summaries to reach a fixed 200 tokens, which manufactures a
request no system ever issues; short targets are instead reported as per-token
acceptance rather than as accepted length.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from specdecode.experiments.sources import Sample, group_histogram, load_samples
from specdecode.scoping.policy import Document, Request

TokenEncoder = Callable[[str], List[int]]


class NotEnoughGroups(ValueError):
    """Raised when a source cannot supply groups that several requests share.

    A group of one collapses the middle rung of the scope ladder onto
    request-local, so the sweep would report three scopes and measure two. Failing
    here is better than reporting that.
    """


def select_groups(
    samples: Sequence[Sample],
    *,
    min_per_group: int,
    max_groups: int,
) -> List[str]:
    """Groups with enough samples to be both history and requests, largest first."""
    counts = group_histogram(samples)
    eligible = [g for g, n in counts.items() if n >= min_per_group]
    eligible.sort(key=lambda g: (-counts[g], g))
    return eligible[:max_groups]


HistoryRows = Dict[str, List[List[int]]]
RequestRows = Dict[str, List[Tuple[List[int], List[int]]]]
Split = Tuple[List[str], HistoryRows, RequestRows]


class _Encoder:
    """Tokenizes a sample into the two shapes the stream needs."""

    def __init__(self, enc: TokenEncoder, doc_token_cap: int, target_token_cap: int) -> None:
        self.enc = enc
        self.doc_cap = doc_token_cap
        self.target_cap = target_token_cap

    def history(self, sample: Sample) -> List[int]:
        # Prompt and response together: retaining only the prompt would make a group
        # scope a pile of questions with none of their answers.
        prompt = self.enc(sample.doc_text)[: self.doc_cap]
        return prompt + self.enc(sample.target_text)[: self.target_cap]

    def request(self, sample: Sample) -> Optional[Tuple[List[int], List[int]]]:
        target = self.enc(sample.target_text)[: self.target_cap]
        return (self.enc(sample.doc_text)[: self.doc_cap], target) if len(target) >= 2 else None


def _split_by_group(
    samples: Sequence[Sample],
    encoder: "_Encoder",
    *,
    history_per_group: int,
    requests_per_group: int,
    max_groups: int,
) -> Split:
    """Within each group, the first samples are history and the next are requests."""
    needed = history_per_group + requests_per_group
    chosen = select_groups(samples, min_per_group=needed, max_groups=max_groups)
    if len(chosen) < 2:
        raise NotEnoughGroups(
            f"only {len(chosen)} group(s) have >= {needed} samples; "
            "the group scope would collapse onto request-local"
        )

    by_group: Dict[str, List[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)

    history_rows: HistoryRows = {}
    request_rows: RequestRows = {}
    for group in chosen:
        bucket = by_group[group]
        history = [t for t in (encoder.history(s) for s in bucket[:history_per_group]) if t]
        tail = bucket[history_per_group : history_per_group + requests_per_group]
        requests = [row for row in (encoder.request(s) for s in tail) if row is not None]
        if history and requests:
            history_rows[group] = history
            request_rows[group] = requests
    return chosen, history_rows, request_rows


def _split_ungrouped(
    samples: Sequence[Sample],
    encoder: "_Encoder",
    *,
    history_per_group: int,
    requests_per_group: int,
    max_groups: int,
) -> Split:
    """A source with no grouping key: split the stream globally, one group per sample.

    The group scope then resolves to an empty datastore, which is the honest answer
    -- there is no group to scope to -- rather than a silent copy of request-local.
    """
    needed = history_per_group + requests_per_group
    # Same history:request ratio as the grouped path, but never past the end of the
    # stream: otherwise a short source spends every sample on history and yields no
    # requests at all.
    n_history = min(
        max_groups * history_per_group,
        max(1, (len(samples) * history_per_group) // needed),
    )
    history_rows: HistoryRows = {}
    request_rows: RequestRows = {}
    for sample in samples[:n_history]:
        tokens = encoder.history(sample)
        if tokens:
            history_rows.setdefault(sample.group_id, []).append(tokens)
    for sample in samples[n_history : n_history + max_groups * requests_per_group]:
        row = encoder.request(sample)
        if row is not None:
            request_rows.setdefault(sample.group_id, []).append(row)

    order = [s.group_id for s in samples][: max_groups * needed]
    chosen = [g for g in order if g in history_rows or g in request_rows]
    return chosen, history_rows, request_rows


def build_stream(
    samples: Sequence[Sample],
    enc: TokenEncoder,
    *,
    history_per_group: int,
    requests_per_group: int,
    doc_token_cap: int,
    target_token_cap: int,
    max_groups: int,
    allow_singleton_groups: bool = False,
) -> Tuple[List[Document], List[Request]]:
    """Split each group's samples into history documents and evaluated requests.

    Within a group the split is by position, so the history a request sees is
    material that genuinely preceded it. Across groups the streams are interleaved
    round-robin so that no group owns a contiguous block of serving order -- with a
    contiguous layout, ``narrow(N)`` would return the same documents as the group
    scope for every early request and the negative control would be silently void.
    """
    encoder = _Encoder(enc, doc_token_cap, target_token_cap)
    split = _split_ungrouped if allow_singleton_groups else _split_by_group
    chosen, history_rows, request_rows = split(
        samples, encoder,
        history_per_group=history_per_group,
        requests_per_group=requests_per_group,
        max_groups=max_groups,
    )

    if not request_rows:
        raise NotEnoughGroups("no group produced both history and requests")

    documents: List[Document] = []
    order = 0
    depth = max((len(v) for v in history_rows.values()), default=0)
    for slot in range(depth):
        for group in chosen:
            bucket = history_rows.get(group, [])
            if slot < len(bucket):
                documents.append(
                    Document(
                        doc_id=f"{group}#h{slot}",
                        group_id=group,
                        order=order,
                        tokens=tuple(bucket[slot]),
                    )
                )
                order += 1

    evaluated: List[Request] = []
    depth = max((len(v) for v in request_rows.values()), default=0)
    for slot in range(depth):
        for group in chosen:
            bucket = request_rows.get(group, [])
            if slot < len(bucket):
                doc, target = bucket[slot]
                evaluated.append(
                    Request(
                        request_id=f"{group}#r{slot}",
                        group_id=group,
                        order=order,
                        doc_tokens=tuple(doc),
                        target_tokens=tuple(target),
                    )
                )
                order += 1

    return documents, evaluated


def stream_key(
    source: str,
    history_per_group: int,
    requests_per_group: int,
    doc_token_cap: int,
    target_token_cap: int,
    max_groups: int,
) -> str:
    return (
        f"{source}__h{history_per_group}_r{requests_per_group}"
        f"_d{doc_token_cap}_t{target_token_cap}_g{max_groups}"
    )


def load_or_build_stream(
    cache_dir: str,
    source: str,
    enc: TokenEncoder,
    *,
    history_per_group: int,
    requests_per_group: int,
    doc_token_cap: int,
    target_token_cap: int,
    max_groups: int,
    sample_limit: int,
    allow_singleton_groups: bool = False,
) -> Tuple[List[Document], List[Request]]:
    """Tokenize once and reuse: the HF download and tokenization dominate runtime."""
    key = stream_key(
        source, history_per_group, requests_per_group,
        doc_token_cap, target_token_cap, max_groups,
    )
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        documents = [
            Document(**{**d, "tokens": tuple(d["tokens"])}) for d in payload["documents"]
        ]
        requests = [
            Request(
                **{
                    **r,
                    "doc_tokens": tuple(r["doc_tokens"]),
                    "target_tokens": tuple(r["target_tokens"]),
                }
            )
            for r in payload["requests"]
        ]
        return documents, requests

    samples = load_samples(source, sample_limit)
    documents, requests = build_stream(
        samples, enc,
        history_per_group=history_per_group,
        requests_per_group=requests_per_group,
        doc_token_cap=doc_token_cap,
        target_token_cap=target_token_cap,
        max_groups=max_groups,
        allow_singleton_groups=allow_singleton_groups,
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "documents": [
                    {
                        "doc_id": d.doc_id,
                        "group_id": d.group_id,
                        "order": d.order,
                        "tokens": list(d.tokens),
                    }
                    for d in documents
                ],
                "requests": [
                    {
                        "request_id": r.request_id,
                        "group_id": r.group_id,
                        "order": r.order,
                        "doc_tokens": list(r.doc_tokens),
                        "target_tokens": list(r.target_tokens),
                    }
                    for r in requests
                ],
            },
            fh,
        )
    return documents, requests
