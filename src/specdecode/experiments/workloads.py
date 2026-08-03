"""Workload construction for the RQ2 factorial.

A workload is a (datastore, targets) pair. The factorial's speculatability axis
is created by a single manipulation, holding everything else fixed:

``pooled``
    the datastore is unrelated material of the same kind
``self_relevant``
    the same datastore size, but each target's own source document is inside it

Probing showed this manipulation separates cleanly where the task copies from its
source (CNN/DailyMail highlights, 2.65x) and barely at all where it rewrites
(XSum, 1.15x) -- so the low/high contrast is a measured property, not an
assumption about which dataset "should" be predictable.

Everything else is held constant: one tokenizer, one target length, one datastore
size per cell. Tokenized workloads are cached on disk, because the Hugging Face
stream is the slowest and least reliable part of the pipeline.
"""

import json
import os
from typing import Callable, Dict, List, Sequence, Tuple

TokenEncoder = Callable[[str], List[int]]
Workload = Tuple[List[int], List[List[int]]]

# Wikipedia is streamed; the rest are ordinary HF splits.
_SUMMARIZATION: Dict[str, Tuple[str, str, str, str]] = {
    # key: (hf name, config, document field, summary field)
    "cnn": ("abisee/cnn_dailymail", "3.0.0", "article", "highlights"),
    "xsum": ("EdinburghNLP/xsum", "", "document", "summary"),
    "samsum": ("knkarthick/samsum", "", "dialogue", "summary"),
}


def _split(name: str, config: str):
    from datasets import load_dataset

    return load_dataset(name, config, split="train") if config else load_dataset(
        name, split="train"
    )


def build_summarization(
    dataset: str,
    variant: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
) -> Workload:
    """Datastore of documents; targets are summaries padded to a fixed length.

    In ``self_relevant`` the document behind each target is written into the
    datastore before the pool is topped up, so the two variants differ only in
    *which* documents are present -- never in how many tokens they hold.
    """
    name, config, doc_field, sum_field = _SUMMARIZATION[dataset]
    ds = _split(name, config)
    pool: List[int] = []
    targets: List[List[int]] = []
    i = 0

    if variant == "self_relevant":
        while len(targets) < n_requests:
            pool.extend(enc(ds[i][doc_field]))
            buf: List[int] = []
            j = i
            while len(buf) < target_tokens:
                buf.extend(enc(ds[j][sum_field]))
                j += 1
            targets.append(buf[:target_tokens])
            i += 1
        while len(pool) < store_tokens:
            pool.extend(enc(ds[i][doc_field]))
            i += 1
    elif variant == "pooled":
        while len(pool) < store_tokens:
            pool.extend(enc(ds[i][doc_field]))
            i += 1
        while len(targets) < n_requests:
            buf = []
            while len(buf) < target_tokens:
                buf.extend(enc(ds[i][sum_field]))
                i += 1
            targets.append(buf[:target_tokens])
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if len(pool) < store_tokens:
        raise ValueError(f"{dataset}/{variant}: only {len(pool)} of {store_tokens} tokens")
    return pool[:store_tokens], targets


def build_wiki(
    variant: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
    lang: str = "en",
) -> Workload:
    """Continuation task. ``self_relevant`` seeds each article's own opening."""
    from specdecode.datasets.wiki import stream_forward_holdout_tokens

    if variant == "pooled":
        datastore, targets, _ = stream_forward_holdout_tokens(
            lang, enc, store_tokens, n_requests, target_tokens,
            min_target_tokens=target_tokens,
        )
        return datastore, targets

    if variant != "self_relevant":
        raise ValueError(f"unknown variant {variant!r}")

    head_budget = n_requests * target_tokens
    datastore, articles, _ = stream_forward_holdout_tokens(
        lang, enc, max(store_tokens - head_budget, target_tokens), n_requests,
        target_tokens * 2, min_target_tokens=target_tokens * 2,
    )
    pool = list(datastore)
    targets: List[List[int]] = []
    for article in articles:
        pool.extend(article[:target_tokens])
        targets.append(article[target_tokens:])
    return pool, targets


def build_workload(
    dataset: str,
    variant: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
) -> Workload:
    if dataset == "wiki":
        return build_wiki(variant, enc, store_tokens, n_requests, target_tokens)
    return build_summarization(
        dataset, variant, enc, store_tokens, n_requests, target_tokens
    )


DATASETS: Sequence[str] = ("cnn", "samsum", "xsum", "wiki")
VARIANTS: Sequence[str] = ("pooled", "self_relevant")


def workload_key(
    dataset: str, variant: str, store_tokens: int, n_requests: int, target_tokens: int
) -> str:
    return f"{dataset}__{variant}__{store_tokens}_{n_requests}_{target_tokens}"


def load_or_build(
    cache_dir: str,
    dataset: str,
    variant: str,
    enc: TokenEncoder,
    store_tokens: int,
    n_requests: int,
    target_tokens: int,
) -> Workload:
    """Tokenize once, reuse forever -- HF streaming is the flakiest step."""
    key = workload_key(dataset, variant, store_tokens, n_requests, target_tokens)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload["datastore"], payload["targets"]

    datastore, targets = build_workload(
        dataset, variant, enc, store_tokens, n_requests, target_tokens
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datastore": datastore, "targets": targets}, fh)
    return datastore, targets
