"""Locate candidate English workloads on the speculatability axis.

RQ2 needs two workloads that differ in how predictable their output is from a
datastore of a *fixed* size. Earlier probing showed two things worth designing
around:

* Datastore **size** is not what drives predictability. Pooling 1000 unrelated
  SQuAD passages scored worse than that request's own single passage.
* A contrast measured across languages (Lao vs English Wikipedia) is a tokenizer
  artifact, not a workload property, so every workload here is English and every
  target is truncated to the same token count.

So each workload is probed in two variants where possible:

``pooled``
    the datastore is unrelated material of the same kind
``self_relevant``
    the same datastore, but the target's own source text is inside it

The gap between those two variants isolates *relevance* from *size*, and the gap
between workloads isolates task type (continuation vs abstractive rewriting).

Results are written incrementally, so a dropped Hugging Face connection costs one
workload rather than the whole run -- re-running skips whatever already finished.

    python scripts/probe_speculatability.py
    python scripts/probe_speculatability.py --datastore-tokens 500000 --n-requests 200
    python scripts/probe_speculatability.py --only wiki_pooled,xsum_pooled --force
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from specdecode.datasets.wiki import stream_forward_holdout_tokens  # noqa: E402
from specdecode.speculatability import (  # noqa: E402
    SuffixStatsIndex,
    profile_with_backoff,
)

Workload = Tuple[List[int], List[List[int]]]  # (datastore tokens, target token lists)


# --------------------------------------------------------------------------- #
# tokenization cache
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def cached(cache_dir: str, key: str, build: Callable[[], Workload]) -> Workload:
    """Build a workload once and reuse it, so re-runs never re-download."""
    path = _cache_path(cache_dir, key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        print(f"  [cache] {key}", flush=True)
        return payload["datastore"], payload["targets"]

    datastore, targets = build()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datastore": datastore, "targets": targets}, fh)
    return datastore, targets


# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #

def _hf_split(name: str, config: str = ""):
    from datasets import load_dataset

    return load_dataset(name, config, split="train") if config else load_dataset(
        name, split="train"
    )


def wiki_pooled(enc, store_tokens: int, n_req: int, tgt_tokens: int) -> Workload:
    """Continuation task. Datastore is other articles; the target's own is absent."""
    datastore, targets, _ = stream_forward_holdout_tokens(
        "en", enc, store_tokens, n_req, tgt_tokens, min_target_tokens=tgt_tokens
    )
    return datastore, targets


def wiki_self_relevant(enc, store_tokens: int, n_req: int, tgt_tokens: int) -> Workload:
    """Same task, but each target's own opening is inside the datastore."""
    head_budget = n_req * tgt_tokens
    datastore, articles, _ = stream_forward_holdout_tokens(
        "en", enc, max(store_tokens - head_budget, tgt_tokens), n_req,
        tgt_tokens * 2, min_target_tokens=tgt_tokens * 2,
    )
    pool = list(datastore)
    targets: List[List[int]] = []
    for article in articles:
        pool.extend(article[:tgt_tokens])       # first half seeds the datastore
        targets.append(article[tgt_tokens:])    # second half is what we "generate"
    return pool, targets


def _summarization(
    name: str, doc_field: str, sum_field: str, config: str = ""
) -> Callable[..., Workload]:
    def pooled(enc, store_tokens: int, n_req: int, tgt_tokens: int) -> Workload:
        ds = _hf_split(name, config)
        pool: List[int] = []
        i = 0
        while len(pool) < store_tokens:
            pool.extend(enc(ds[i][doc_field]))
            i += 1
        targets: List[List[int]] = []
        while len(targets) < n_req:             # concatenate summaries to a fixed length
            buf: List[int] = []
            while len(buf) < tgt_tokens:
                buf.extend(enc(ds[i][sum_field]))
                i += 1
            targets.append(buf[:tgt_tokens])
        return pool[:store_tokens], targets

    return pooled


def _summarization_self_relevant(
    name: str, doc_field: str, sum_field: str, config: str = ""
) -> Callable[..., Workload]:
    def self_relevant(enc, store_tokens: int, n_req: int, tgt_tokens: int) -> Workload:
        ds = _hf_split(name, config)
        pool: List[int] = []
        targets: List[List[int]] = []
        i = 0
        while len(targets) < n_req:
            pool.extend(enc(ds[i][doc_field]))  # the source document goes in
            buf: List[int] = []
            j = i
            while len(buf) < tgt_tokens:
                buf.extend(enc(ds[j][sum_field]))
                j += 1
            targets.append(buf[:tgt_tokens])
            i += 1
        while len(pool) < store_tokens:
            pool.extend(enc(ds[i][doc_field]))
            i += 1
        return pool[:store_tokens], targets

    return self_relevant


WORKLOADS: Dict[str, Callable[..., Workload]] = {
    "wiki_pooled": wiki_pooled,
    "wiki_self_relevant": wiki_self_relevant,
    "xsum_pooled": _summarization("EdinburghNLP/xsum", "document", "summary"),
    "xsum_self_relevant": _summarization_self_relevant(
        "EdinburghNLP/xsum", "document", "summary"
    ),
    "samsum_pooled": _summarization("knkarthick/samsum", "dialogue", "summary"),
    "samsum_self_relevant": _summarization_self_relevant(
        "knkarthick/samsum", "dialogue", "summary"
    ),
    "cnn_pooled": _summarization(
        "abisee/cnn_dailymail", "article", "highlights", config="3.0.0"
    ),
    "cnn_self_relevant": _summarization_self_relevant(
        "abisee/cnn_dailymail", "article", "highlights", config="3.0.0"
    ),
}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def measure(datastore: Sequence[int], targets: Sequence[Sequence[int]], max_k: int) -> Dict:
    index = SuffixStatsIndex(datastore, max_k=max_k)
    profiles = [profile_with_backoff(index, t) for t in targets]
    oracle = [p.oracle_accept_rate for p in profiles]
    return {
        "n_requests": len(profiles),
        "datastore_tokens": len(datastore),
        "target_tokens": len(targets[0]) if targets else 0,
        "oracle_q1": quantile(oracle, 0.25),
        "oracle_median": quantile(oracle, 0.5),
        "oracle_q3": quantile(oracle, 0.75),
        "oracle_mean": statistics.fmean(oracle),
        "matched_k": statistics.fmean(p.mean_matched_k for p in profiles),
        "coverage": statistics.fmean(p.coverage for p in profiles),
        "mean_support": statistics.fmean(p.mean_support for p in profiles),
        "branch_entropy_bits": statistics.fmean(p.mean_branch_entropy for p in profiles),
    }


def load_results(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("workloads", {})


def save_results(path: str, config: Dict, results: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"config": config, "workloads": results}, fh, indent=2)


def print_table(results: Dict[str, Dict]) -> None:
    header = (
        f"{'workload':>22} | {'store':>9} | {'oracle Q1':>9} | {'median':>7} | "
        f"{'Q3':>6} | {'match k':>7} | {'cover':>5} | {'support':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].get("oracle_median", 0)):
        if "error" in r:
            print(f"{name:>22} | FAILED: {r['error'][:60]}")
            continue
        print(
            f"{name:>22} | {r['datastore_tokens']:>9,} | {r['oracle_q1']:>9.3f} | "
            f"{r['oracle_median']:>7.3f} | {r['oracle_q3']:>6.3f} | "
            f"{r['matched_k']:>7.2f} | {r['coverage']:>5.3f} | {r['mean_support']:>8.0f}"
        )

    ok = {k: v for k, v in results.items() if "oracle_median" in v}
    if len(ok) >= 2:
        best = max(ok.items(), key=lambda kv: kv[1]["oracle_median"])
        worst = min(ok.items(), key=lambda kv: kv[1]["oracle_median"])
        ratio = best[1]["oracle_median"] / max(worst[1]["oracle_median"], 1e-9)
        print(
            f"\nwidest separation: {best[0]} / {worst[0]} = {ratio:.2f}x"
            f"   ({best[1]['oracle_median']:.3f} vs {worst[1]['oracle_median']:.3f})"
        )
        print("a usable RQ2 pair needs roughly 2x or more, with Q1/Q3 ranges that do not overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--datastore-tokens", type=int, default=200_000)
    parser.add_argument("--n-requests", type=int, default=120)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--only", default="", help="comma-separated workload names")
    parser.add_argument("--force", action="store_true", help="recompute finished workloads")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_workload_probe", "probe.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_workload_probe", "cache"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text)

    selected = [w.strip() for w in args.only.split(",") if w.strip()] or list(WORKLOADS)
    unknown = [w for w in selected if w not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workloads: {unknown}. choose from {list(WORKLOADS)}")

    config = {
        "tokenizer": args.tokenizer,
        "datastore_tokens": args.datastore_tokens,
        "n_requests": args.n_requests,
        "target_tokens": args.target_tokens,
        "max_k": args.max_k,
    }
    results = {} if args.force else load_results(args.output)

    print(f"tokenizer         {args.tokenizer}")
    print(f"datastore tokens  {args.datastore_tokens:,}")
    print(f"requests          {args.n_requests}   target tokens {args.target_tokens}")
    print(f"output            {args.output}\n")

    for name in selected:
        if name in results and "error" not in results[name]:
            print(f"[skip] {name} (already done)", flush=True)
            continue

        print(f"[run ] {name} ...", flush=True)
        started = time.time()
        try:
            key = f"{name}__{args.datastore_tokens}_{args.n_requests}_{args.target_tokens}"
            datastore, targets = cached(
                args.cache_dir,
                key,
                lambda: WORKLOADS[name](  # noqa: B023
                    enc, args.datastore_tokens, args.n_requests, args.target_tokens
                ),
            )
            results[name] = measure(datastore, targets, args.max_k)
            print(
                f"       oracle median {results[name]['oracle_median']:.3f}"
                f"   ({time.time() - started:.0f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad workload must not kill the run
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"       FAILED: {type(exc).__name__}: {exc}", flush=True)

        save_results(args.output, config, results)  # checkpoint after every workload

    print_table(results)
    print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
