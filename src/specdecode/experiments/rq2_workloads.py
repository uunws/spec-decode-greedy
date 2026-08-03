"""Workload construction for the RQ2 factorial, with the dataset as the factor.

RQ2's speculatability axis is **which workload this is**, not what we did to its
datastore. That distinction is what keeps RQ2 and RQ3 from collapsing into one
experiment: RQ3 holds the dataset fixed and reshapes the datastore, so if RQ2 also
varied datastore relevance the two would be measuring the same manipulation twice.

    RQ2   datastore built the same way everywhere, dataset varies
    RQ3   dataset fixed, datastore scope varies

The construction below is therefore identical for every dataset::

    datastore = documents concatenated, truncated to exactly ``store_tokens``
    targets   = ``n_requests`` model outputs, each capped at ``target_tokens``
    holdout   = a target's own document is never in the datastore

The holdout matters. Leaving a target's source document in would plant a copy of
the answer and measure the planting, not the workload; what is left is the
repetition a domain has on its own, which is what "workload character" means.

Selection rule, stated before any measurement: a dataset is a candidate only if its
natural outputs reach ``min_target_tokens``. ``L/S`` has no range when the target is
shorter than a draft, so short-output datasets cannot carry a speedup comparison --
this excludes SQuAD (~3 tokens) by construction rather than by result. Nothing is
ever excluded for scoring badly.

The threshold was 64 in the first pass and is now 24. That change was made after
seeing the first classification, so it is recorded here rather than quietly edited:
at 64 the rule excluded XSum and SAMSum, which is to say every abstractive
summarizer, which is to say the entire low-support/low-structure corner the
factorial needs a low level from. The remaining candidates were all code, chat or
news and clustered together. The revision is about the rule deleting a region of
the design space, not about any dataset's score -- where XSum and SAMSum land was
unknown when it was made, and they stay in the set wherever they land.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from specdecode.experiments.sources import Sample, load_samples

TokenEncoder = Callable[[str], List[int]]

# Long enough for accepted length to have somewhere to go. A 24-token target still
# spans roughly six to twenty-four speculative steps at B=4, so L/S can move; below
# that the request is shorter than a couple of drafts and the ratio is pinned.
MIN_TARGET_TOKENS = 24

# The candidate set, chosen to span the plane rather than to win. Two are expected
# at each corner and the expectation is written down here so the measurement can
# contradict it: code and maths for structure, chat and agent traces in between,
# abstractive summarization and open instructions for the low corner.
CANDIDATES: Sequence[str] = (
    "codesearch", "gsm8k", "swetraj", "wildchat", "cnn", "samsum", "xsum", "dolly",
)


class NotEnoughData(RuntimeError):
    """A source could not supply the requested datastore or request count."""


@dataclass(frozen=True)
class Rq2Workload:
    """One dataset rendered into the shape every cell of the factorial expects."""

    dataset: str
    datastore: List[int]
    targets: List[List[int]]
    n_docs_in_store: int
    mean_target_tokens: float
    samples_scanned: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "store_tokens": len(self.datastore),
            "n_requests": len(self.targets),
            "n_docs_in_store": self.n_docs_in_store,
            "mean_target_tokens": self.mean_target_tokens,
            "samples_scanned": self.samples_scanned,
        }


def _encode_all(samples: Sequence[Sample], enc: TokenEncoder) -> List[Dict[str, List[int]]]:
    return [{"doc": enc(s.doc_text), "target": enc(s.target_text)} for s in samples]


def build_workload(
    dataset: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
    *,
    sample_limit: int = 0,
    min_target_tokens: int = MIN_TARGET_TOKENS,
) -> Rq2Workload:
    """Datastore of exactly ``store_tokens``, plus ``n_requests`` held-out targets.

    Requests are taken from the *front* of the stream and their documents are
    skipped; the datastore is then filled from what follows. Taking requests first
    is what guarantees the holdout without having to search for collisions later.

    Targets are truncated to ``target_tokens`` but never padded. Concatenating
    several short outputs to reach a length quota manufactures a request no system
    would ever issue, which is a mistake this repo has already made once.
    """
    limit = sample_limit or max(n_requests * 4, 4000)
    samples = load_samples(dataset, limit)
    encoded = _encode_all(samples, enc)

    targets: List[List[int]] = []
    consumed = 0
    for item in encoded:
        if len(targets) >= n_requests:
            break
        consumed += 1
        if len(item["target"]) < min_target_tokens:
            continue
        targets.append(item["target"][:target_tokens])

    if len(targets) < n_requests:
        raise NotEnoughData(
            f"{dataset}: only {len(targets)} of {n_requests} targets reach "
            f"{min_target_tokens} tokens in {len(encoded)} samples"
        )

    datastore: List[int] = []
    n_docs = 0
    for item in encoded[consumed:]:
        if len(datastore) >= store_tokens:
            break
        datastore.extend(item["doc"])
        datastore.extend(item["target"])
        n_docs += 1

    if len(datastore) < store_tokens:
        raise NotEnoughData(
            f"{dataset}: datastore reached {len(datastore)} of {store_tokens} tokens; "
            f"raise --sample-limit"
        )

    return Rq2Workload(
        dataset=dataset,
        datastore=datastore[:store_tokens],
        targets=targets,
        n_docs_in_store=n_docs,
        mean_target_tokens=sum(len(t) for t in targets) / len(targets),
        samples_scanned=len(encoded),
    )


def workload_key(dataset: str, store_tokens: int, n_requests: int, target_tokens: int) -> str:
    return f"rq2__{dataset}__{store_tokens}_{n_requests}_{target_tokens}"


def load_or_build(
    cache_dir: str,
    dataset: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
    *,
    sample_limit: int = 0,
) -> Rq2Workload:
    """Tokenize once, reuse forever. HF streaming is the flakiest step in the run."""
    key = workload_key(dataset, store_tokens, n_requests, target_tokens)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return Rq2Workload(
            dataset=dataset,
            datastore=payload["datastore"],
            targets=payload["targets"],
            n_docs_in_store=payload.get("n_docs_in_store", 0),
            mean_target_tokens=payload.get("mean_target_tokens", 0.0),
            samples_scanned=payload.get("samples_scanned", 0),
        )

    workload = build_workload(
        dataset, enc, store_tokens, n_requests, target_tokens, sample_limit=sample_limit
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "datastore": workload.datastore,
                "targets": workload.targets,
                "n_docs_in_store": workload.n_docs_in_store,
                "mean_target_tokens": workload.mean_target_tokens,
                "samples_scanned": workload.samples_scanned,
            },
            fh,
        )
    return workload
