"""RQ3: can history scoping raise workload speculatability without touching the algorithm?

What varies and what does not
-----------------------------
One drafter class, one set of parameters, one cost model, for every cell. The only
thing that changes is which tokens were used to build the index. That is what makes
"without changing the underlying algorithm" a statement about the experiment rather
than a claim in the discussion.

    global   every group's history          REST
    group    the request's own group only   DAS / per-problem
    local    the request's prompt only      Fast Context
    narrow_N the N most recent same-group   negative control: cuts on recency

Why every treatment has a size-matched control
----------------------------------------------
Narrowing a scope moves two things at once and they oppose each other: relevance
rises, corpus size falls. A bare scope comparison therefore cannot be read -- a flat
result is equally consistent with "relevance does nothing" and with "relevance does
a great deal and small corpora cost exactly as much". Each treatment is paired with
a random datastore of the same token count, and the gap between them is the part
attributable to relevance and nothing else.

Metrics
-------
Speculatability is reported as the two axes the paper defines, never collapsed into
one number: ``mean_support`` (History) and ``1 - normalized_branch_entropy``
(Structure). No composite is used, so RQ3 carries no dependency on RQ1. The two move
in opposite directions when a scope narrows, so the plane alone cannot rank scopes;
``L/S`` and the modelled speedup are the tiebreakers and are reported beside them,
never as an axis.

    python scripts/run_rq3.py --dry-run
    python scripts/run_rq3.py --sources squad --n-grams 3
    python scripts/run_rq3.py
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from specdecode.costmodel import (  # noqa: E402
    DrafterCostProfile,
    TargetLatencyModel,
    WorkloadCounts,
    speedup_both_ways,
)
from specdecode.experiments.rq3_workloads import load_or_build_stream  # noqa: E402
from specdecode.experiments.sources import ALL_SOURCES, SPECS  # noqa: E402
from specdecode.experiments.timing import (  # noqa: E402
    measure_timer_overhead_ns,
    pinned_threads,
    warmup,
)
from specdecode.scoping import (  # noqa: E402
    Document,
    GlobalScope,
    GroupScoped,
    NarrowScoped,
    Request,
    RequestLocal,
    ScopedIndexCache,
    ScopePolicy,
    SizeMatchedControl,
    assert_causal,
    assert_key_determines_tokens,
    group_sizes,
)
from specdecode.simulator.drafter.tensorNGramDrafter import (  # noqa: E402
    IndexedTensorNGramDrafter,
)
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics  # noqa: E402
from specdecode.simulator.playback.speculativePlayback import (  # noqa: E402
    TensorSpeculativePlayback,
)
from specdecode.simulator.verifier.tensorGreedyVerifier import (  # noqa: E402
    TensorGreedyVerifier,
)
from specdecode.speculatability.index import SuffixStatsIndex  # noqa: E402
from specdecode.speculatability.profile import profile_with_backoff  # noqa: E402

# The same assumed target profile RQ2 uses, so the two sit on one axis. No real
# model is run: source is "assumed", never "calibrated", and every number derived
# from it is reported as a threshold rather than a value.
DEFAULT_TARGET = TargetLatencyModel(
    t_verify_1_ns=20_000_000,
    beta=0.01,
    source="assumed",
    model_name="assumed-7b",
    hardware="assumed-gpu",
)

# Metrics averaged across requests when a cell is summarised.
METRIC_KEYS = (
    "corpus_tokens", "reuse", "build_ns", "build_bytes",
    "median_draft_ns", "p95_draft_ns", "rho",
    "mean_support", "median_support", "mean_branch_entropy",
    "normalized_branch_entropy", "mean_branch_factor", "coverage",
    "mean_matched_k", "oracle_accept_rate",
)


def make_policy(name: str, seed: int) -> ScopePolicy:
    if name == "local":
        return RequestLocal()
    if name == "group":
        return GroupScoped()
    if name == "global":
        return GlobalScope()
    if name.startswith("narrow"):
        return NarrowScoped(int(name[len("narrow") :]))
    if name.startswith("control_"):
        return SizeMatchedControl(
            treatment=make_policy(name[len("control_") :], seed), seed=seed
        )
    raise ValueError(f"unknown scope {name!r}")


def group_by_key(
    policy: ScopePolicy, requests: Sequence[Request], history: Sequence[Document]
) -> List[Tuple[str, List[int], List[Request]]]:
    """Requests resolving to the same datastore, so one built index serves them all."""
    buckets: Dict[str, Tuple[List[int], List[Request]]] = {}
    order: List[str] = []
    for request in requests:
        scope = policy.resolve(request, history)
        if scope.key not in buckets:
            buckets[scope.key] = (scope.tokens, [])
            order.append(scope.key)
        buckets[scope.key][1].append(request)
    return [(k, buckets[k][0], buckets[k][1]) for k in order]


def run_accounting(
    drafter: object, targets: Sequence[Sequence[int]]
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Token accounting, pooled and per request.

    Per request as well as pooled because the global scope puts every group in one
    bucket: a pooled ``L/S`` would be reported identically for every group and the
    per-group breakdown would say nothing.
    """
    verifier = TensorGreedyVerifier()
    total_l = total_s = accepted = rejected = 0
    ratios: List[float] = []
    rows: List[Dict[str, float]] = []

    for target in targets:
        metrics = PlaybackMetrics()
        playback = TensorSpeculativePlayback(None, drafter, verifier, metrics)  # type: ignore[arg-type]
        if playback.run_tokens(list(target)) != list(target):
            raise RuntimeError("playback did not reconstruct the target losslessly")
        total_l += metrics.normal_steps
        total_s += metrics.speculative_steps
        accepted += metrics.accepted_tokens
        rejected += metrics.rejected_tokens
        if metrics.speculative_steps:
            ratios.append(metrics.normal_steps / metrics.speculative_steps)
        rows.append({
            "normal_steps": float(metrics.normal_steps),
            "speculative_steps": float(metrics.speculative_steps),
            "accepted": float(metrics.accepted_tokens),
            "rejected": float(metrics.rejected_tokens),
        })

    pooled = {
        "target_tokens": total_l,
        "speculative_steps": total_s,
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "acceptance_rate": accepted / max(accepted + rejected, 1),
        # Accepted tokens per target token. Unlike L/S this stays meaningful when a
        # target is shorter than the draft depth, which is the case for SQuAD (~3
        # tokens), Spider (~30) and SAMSum (~25).
        "accepted_per_token": accepted / max(total_l, 1),
        "token_speedup": total_l / max(total_s, 1),
        "token_speedup_median": statistics.median(ratios) if ratios else 1.0,
        "n_requests": len(targets),
    }
    return pooled, rows


def measure_latency(
    drafter: object, prompts: Sequence[Sequence[int]], *, budget_s: float, min_calls: int
) -> Dict[str, float]:
    """Median and p95 ``generate_draft`` cost.

    p95 is carried because RQ2 found the median can *fall* as a datastore grows: a
    scanning drafter returns on its first match, so a bigger corpus finds one sooner
    and only the no-match tail shows the true scaling.
    """
    samples: List[int] = []
    deadline = time.perf_counter() + budget_s
    while True:
        for prompt in prompts:
            start = time.perf_counter_ns()
            drafter.generate_draft(list(prompt))  # type: ignore[attr-defined]
            samples.append(time.perf_counter_ns() - start)
        if len(samples) >= min_calls and time.perf_counter() > deadline:
            break
        if len(samples) > 20 * min_calls:
            break
    samples.sort()
    return {
        "median_ns": float(statistics.median(samples)),
        "mean_ns": float(statistics.fmean(samples)),
        "p95_ns": float(samples[int(0.95 * (len(samples) - 1))]),
        "calls": len(samples),
    }


def prompts_from(requests: Sequence[Request], n_gram: int, limit: int) -> List[List[int]]:
    """Prefixes the drafter is actually queried with during generation."""
    out: List[List[int]] = []
    span = max(1, n_gram - 1)
    for request in requests:
        target = request.target_tokens
        step = max(1, len(target) // 8)
        for pos in range(span, len(target), step):
            out.append(list(target[pos - span : pos]))
            if len(out) >= limit:
                return out
    return out or [[0] * span]


def speculatability(
    corpus: Sequence[int], requests: Sequence[Request], max_k: int, progress: str
) -> List[Dict[str, float]]:
    """The two axes, per request, measured against one datastore.

    ``mean_support`` and ``normalized_branch_entropy`` are the paper's History and
    Structure axes and are the primary report. ``coverage`` travels with them as a
    mandatory caveat: support is averaged over covered positions only, so a scope
    that finds almost nothing can post a flattering support number.

    ``oracle_accept_rate`` is recorded but is not a headline -- it is kept only so
    that a disagreement between the plane and real predictability is visible rather
    than invisible.
    """
    print(f"        {progress} suffix stats over {len(corpus):,} tokens", flush=True)
    index = SuffixStatsIndex(list(corpus), max_k=max_k)
    out: List[Dict[str, float]] = []
    for request in requests:
        profile = profile_with_backoff(index, list(request.target_tokens))
        out.append({
            "mean_support": profile.mean_support,
            "median_support": profile.median_support,
            "mean_branch_entropy": profile.mean_branch_entropy,
            "normalized_branch_entropy": profile.normalized_branch_entropy,
            "mean_branch_factor": profile.mean_branch_factor,
            "coverage": profile.coverage,
            "mean_matched_k": profile.mean_matched_k,
            "oracle_accept_rate": profile.oracle_accept_rate,
        })
    return out


def _mean(rows: Sequence[Dict], key: str) -> float:
    values = [r[key] for r in rows if key in r]
    return statistics.fmean(values) if values else 0.0


def _total(rows: Sequence[Dict], key: str) -> float:
    return sum(r.get(key, 0.0) for r in rows)


def aggregate(rows: Sequence[Dict], beta_budget: float, t_verify_1_ns: float) -> Dict:
    """Pool step counts before dividing, exactly as RQ2 does.

    Averaging per-request ratios would weight a three-token SQuAD answer as heavily
    as a 500-token chat turn. Both speedups come from the same pooled counts, so
    ``including_build`` can never exceed ``excluding_build`` -- an inversion a mixed
    aggregation produces and that would be nonsense.
    """
    out = {k: _mean(rows, k) for k in METRIC_KEYS}
    total_l = _total(rows, "normal_steps")
    spec = _total(rows, "speculative_steps")
    acc, rej = _total(rows, "accepted"), _total(rows, "rejected")

    out["normal_steps"] = total_l
    out["speculative_steps"] = spec
    out["token_speedup"] = total_l / max(spec, 1.0)
    out["accepted_per_step"] = acc / max(spec, 1.0)
    out["accepted_per_token"] = acc / max(total_l, 1.0)
    out["acceptance_rate"] = acc / max(acc + rej, 1.0)
    out["mean_target_tokens"] = total_l / max(len(rows), 1)

    # The Structure axis. Entropy is "bad up", so the plane plots its complement and
    # the sign convention is fixed here rather than in each figure.
    out["structure"] = 1.0 - out["normalized_branch_entropy"]

    step_cost = 1.0 + out["rho"] + beta_budget
    out["speedup_excluding_build"] = out["token_speedup"] / step_cost
    # Each row carries its bucket's build cost and how many requests shared it, so
    # dividing before summing amortizes each build exactly once.
    build_steps = sum(
        r.get("build_ns", 0.0) / max(r.get("reuse", 1.0), 1.0) for r in rows
    ) / t_verify_1_ns
    out["build_step_equivalents"] = build_steps
    out["speedup_including_build"] = total_l / max(spec * step_cost + build_steps, 1e-9)
    return out


def run_scope(  # noqa: C901 - one linear pipeline, clearer unsplit
    policy: ScopePolicy,
    documents: Sequence[Document],
    requests: Sequence[Request],
    args,
    n_gram: int,
    timer_overhead: float,
) -> Dict:
    checked = assert_causal(policy, requests, documents)
    n_keys = assert_key_determines_tokens(policy, requests, documents)
    buckets = group_by_key(policy, requests, documents)
    assert n_keys == len(buckets)
    print(f"      causality ok on {checked} requests | {len(buckets)} datastore(s)", flush=True)

    cache = ScopedIndexCache(max_k=max(1, n_gram - 1), capacity=1)
    per_request_rows: List[Dict] = []
    corpus_sizes: List[int] = []

    for b, (key, tokens, bucket_requests) in enumerate(buckets, 1):
        tag = f"[{b}/{len(buckets)}]"
        corpus_sizes.append(len(tokens))
        index, built = cache.get(key, tokens)
        build_ns, build_bytes = cache.build_cost(key)
        if built and len(buckets) <= 24:
            print(
                f"        {tag} built {key[:40]:<40} {len(tokens):>9,} tok "
                f"{build_ns / 1e6:>7.0f} ms",
                flush=True,
            )

        drafter = IndexedTensorNGramDrafter(
            index=index, n=n_gram,
            num_sequences=args.num_sequences, draft_depth=args.budget,
        )

        with pinned_threads(1):
            accounting, accounting_rows = run_accounting(
                drafter, [r.target_tokens for r in bucket_requests]
            )
            probes = prompts_from(bucket_requests, n_gram, args.latency_prompts)
            warmup(drafter, probes[:32], rounds=2)
            latency = measure_latency(
                drafter, probes,
                budget_s=args.latency_budget_s, min_calls=args.latency_min_calls,
            )

        spec = (
            speculatability(tokens, bucket_requests, max(1, n_gram - 1), tag)
            if not args.skip_speculatability
            else [{} for _ in bucket_requests]
        )

        reuse = len(bucket_requests)
        cache.record_requests(reuse)
        counts = WorkloadCounts(
            target_tokens=int(accounting["target_tokens"]),
            speculative_steps=int(accounting["speculative_steps"]),
            draft_budget_tokens=args.num_sequences * args.budget,
        )
        profile = DrafterCostProfile(
            median_draft_ns=latency["median_ns"],
            mean_draft_ns=latency["mean_ns"],
            p95_draft_ns=latency["p95_ns"],
            calls=int(latency["calls"]),
            build_ns=build_ns, build_bytes=build_bytes,
            timer_overhead_ns=timer_overhead,
        )
        both = speedup_both_ways(counts, profile, DEFAULT_TARGET, amortization_requests=reuse)

        for request, request_spec, acc in zip(bucket_requests, spec, accounting_rows):
            per_request_rows.append({
                "group_id": request.group_id,
                "scope_key": key,
                "corpus_tokens": len(tokens),
                "reuse": reuse,
                "build_ns": build_ns,
                "build_bytes": build_bytes,
                "median_draft_ns": latency["median_ns"],
                "p95_draft_ns": latency["p95_ns"],
                "rho": both["excluding_build"].rho,
                **acc,
                **request_spec,
            })

    beta_budget = DEFAULT_TARGET.beta * args.num_sequences * args.budget
    t_v1 = float(DEFAULT_TARGET.t_verify_1_ns)
    groups = sorted({r["group_id"] for r in per_request_rows})
    per_group = {
        g: aggregate([r for r in per_request_rows if r["group_id"] == g], beta_budget, t_v1)
        for g in groups
    } if len(groups) <= 40 else {}

    return {
        "policy": policy.name,
        "n_requests": len(per_request_rows),
        "n_datastores": len(buckets),
        "n_groups": len(groups),
        "causality_checked": checked,
        "mean_corpus_tokens": statistics.fmean(corpus_sizes) if corpus_sizes else 0.0,
        "overall": aggregate(per_request_rows, beta_budget, t_v1),
        "per_group": per_group,
        "cache": cache.stats().as_dict(),
    }


def check_size_match(cell: Dict, control: Dict, tolerance: float) -> Optional[str]:
    """A control that is not the treatment's size is not a control.

    Reported rather than raised: whole documents overshoot at small budgets, and a
    cell that fails this is still worth keeping as long as the failure is visible
    next to the number it invalidates.
    """
    treat = cell["overall"]["corpus_tokens"]
    ctrl = control["overall"]["corpus_tokens"]
    if treat <= 0:
        return "treatment has an empty datastore"
    drift = abs(ctrl - treat) / treat
    if drift > tolerance:
        return f"control is {drift:.1%} off the treatment size ({ctrl:,.0f} vs {treat:,.0f})"
    return None


def summarise_cell(cell: Dict) -> str:
    o = cell["overall"]
    return (
        f"corpus {cell['mean_corpus_tokens']:>9,.0f} tok  "
        f"support {o['mean_support']:>8.1f}  structure {o['structure']:.3f}  "
        f"cov {o['coverage']:.3f}  L/S {o['token_speedup']:.3f}  "
        f"acc/tok {o['accepted_per_token']:.3f}  R={o['reuse']:.0f}  "
        f"speedup {o['speedup_excluding_build']:.4f}"
        f" (incl build {o['speedup_including_build']:.4f})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--scopes", default="global,group,narrow1,narrow5,narrow10,local")
    parser.add_argument("--controls", default="global,group,local",
                        help="treatments that also get a size-matched random control")
    parser.add_argument("--n-grams", default="3", help="n-gram orders to sweep, e.g. 2,3,4")
    parser.add_argument("--history-per-group", type=int, default=40)
    parser.add_argument("--requests-per-group", type=int, default=10)
    parser.add_argument("--max-groups", type=int, default=20)
    parser.add_argument("--sample-limit", type=int, default=40_000)
    parser.add_argument("--doc-token-cap", type=int, default=1024)
    parser.add_argument("--target-token-cap", type=int, default=512)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--latency-prompts", type=int, default=200)
    parser.add_argument("--latency-budget-s", type=float, default=2.0)
    parser.add_argument("--latency-min-calls", type=int, default=200)
    parser.add_argument("--size-match-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-speculatability", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq3_scoping", "results.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq3_scoping", "cache"),
    )
    return parser


def main() -> None:  # noqa: C901
    args = build_parser().parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    n_grams = [int(n) for n in args.n_grams.split(",") if n.strip()]

    print("=" * 78)
    print("RQ3  history scoping")
    print("=" * 78)
    for name in sources:
        spec = SPECS.get(name)
        if spec:
            print(f"  {name:<11} group={spec.group_field or '(none)':<16} {spec.note}")
    print(f"\nscopes       {scopes}")
    print(f"controls     {controls}")
    print(f"n-gram       {n_grams}")
    print(f"stream       {args.history_per_group} history + {args.requests_per_group} "
          f"requests per group, up to {args.max_groups} groups")
    print(f"budget B     {args.budget}   num_sequences={args.num_sequences}")
    print(f"output       {args.output}")
    print("=" * 78, flush=True)

    total_cells = len(sources) * len(n_grams) * (len(scopes) + len(controls))
    if args.dry_run:
        for source in sources:
            for n in n_grams:
                for scope in scopes:
                    mark = "  (+ control)" if scope in controls else ""
                    print(f"  {source}|n{n}|{scope}{mark}")
        print(f"\ndry run: {total_cells} cells, nothing executed")
        return

    from transformers import AutoTokenizer

    print("\nloading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text)

    results: Dict[str, Dict] = {}
    previous_streams: Dict[str, Dict] = {}
    if os.path.exists(args.output) and not args.force:
        with open(args.output, encoding="utf-8") as fh:
            payload = json.load(fh)
        results = payload.get("cells", {})
        # Carry forward stream metadata for sources this invocation is not running.
        # It is not decoration: the report picks L/S or per-token acceptance from
        # `median_target_tokens`, so dropping it silently changes which metric a
        # previously measured cell is reported under.
        previous_streams = payload.get("config", {}).get("streams", {}) or {}
        print(f"resuming: {len(results)} cells already present", flush=True)

    timer_overhead = measure_timer_overhead_ns()
    config = {
        "tokenizer": args.tokenizer,
        "sources": sources,
        "scopes": scopes,
        "controls": controls,
        "n_grams": n_grams,
        "history_per_group": args.history_per_group,
        "requests_per_group": args.requests_per_group,
        "max_groups": args.max_groups,
        "doc_token_cap": args.doc_token_cap,
        "target_token_cap": args.target_token_cap,
        "budget": args.budget,
        "num_sequences": args.num_sequences,
        "seed": args.seed,
        "timer_overhead_ns": timer_overhead,
        "target_latency_model": DEFAULT_TARGET.as_dict(),
        "streams": dict(previous_streams),
    }

    def save() -> None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"config": config, "cells": results}, fh, indent=2)

    started = time.time()
    done = 0

    for source in sources:
        print(f"\n{'=' * 78}\nsource {source}\n{'=' * 78}", flush=True)
        try:
            t0 = time.time()
            documents, requests = load_or_build_stream(
                args.cache_dir, source, enc,
                history_per_group=args.history_per_group,
                requests_per_group=args.requests_per_group,
                doc_token_cap=args.doc_token_cap,
                target_token_cap=args.target_token_cap,
                max_groups=args.max_groups,
                sample_limit=args.sample_limit,
                allow_singleton_groups=(SPECS[source].grouping == "none"),
            )
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the sweep
            print(f"  SKIP {source}: {type(exc).__name__}: {exc}", flush=True)
            config["streams"][source] = {"error": f"{type(exc).__name__}: {exc}"}
            save()
            continue

        sizes = group_sizes(documents)
        history_tokens = sum(len(d) for d in documents)
        target_lengths = [len(r.target_tokens) for r in requests]
        config["streams"][source] = {
            "n_documents": len(documents),
            "history_tokens": history_tokens,
            "n_requests": len(requests),
            "n_groups": len(sizes),
            "median_group_documents": statistics.median(sizes.values()) if sizes else 0,
            "median_target_tokens": statistics.median(target_lengths) if target_lengths else 0,
            "grouping": SPECS[source].grouping,
        }
        print(
            f"  {len(documents):,} history docs ({history_tokens:,} tok) | "
            f"{len(requests):,} requests | {len(sizes)} groups | "
            f"median target {statistics.median(target_lengths):.0f} tok "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        save()

        for n_gram in n_grams:
            plan = [(s, make_policy(s, args.seed)) for s in scopes]
            plan += [
                (f"control_{s}", SizeMatchedControl(make_policy(s, args.seed), seed=args.seed))
                for s in controls if s in scopes
            ]
            for scope_name, policy in plan:
                cell_id = f"{source}|n{n_gram}|{scope_name}"
                done += 1
                if cell_id in results and "error" not in results[cell_id] and not args.force:
                    print(f"  [{done}/{total_cells}] skip {cell_id}", flush=True)
                    continue
                print(f"  [{done}/{total_cells}] {cell_id}", flush=True)
                t_cell = time.time()
                try:
                    cell = run_scope(
                        policy, documents, requests, args, n_gram, timer_overhead
                    )
                    cell.update({
                        "source": source, "scope": scope_name, "n_gram": n_gram,
                        "seconds": time.time() - t_cell,
                    })
                    if scope_name.startswith("control_"):
                        cell["control_for"] = scope_name[len("control_") :]
                    results[cell_id] = cell
                    print(f"      -> {summarise_cell(cell)}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    results[cell_id] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"      FAILED: {type(exc).__name__}: {exc}", flush=True)
                save()

    # Size-match audit last, once every treatment and control exists.
    warnings: List[str] = []
    for cell_id, cell in results.items():
        if "control_for" not in cell:
            continue
        treatment_id = cell_id.replace("|control_", "|")
        treatment = results.get(treatment_id)
        if not treatment or "error" in treatment:
            continue
        problem = check_size_match(treatment, cell, args.size_match_tolerance)
        if problem:
            warnings.append(f"{cell_id}: {problem}")
    config["size_match_warnings"] = warnings
    save()

    ok = sum(1 for r in results.values() if "error" not in r)
    print("\n" + "=" * 78)
    print(f"done: {ok}/{len(results)} cells in {(time.time() - started) / 60:.1f} min")
    if warnings:
        print(f"\n{len(warnings)} size-match warning(s):")
        for line in warnings:
            print(f"  {line}")
    print(f"\nresults written to {args.output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
