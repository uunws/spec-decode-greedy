"""RQ2 factorial: workload speculatability x speculation efficiency -> speedup.

Design
------
Two factors, each with two levels, and everything else held fixed:

    workload    low-speculatability dataset  vs  high-speculatability dataset
    efficiency  on-the-fly retrieval         vs  precomputed lookup

The workload levels come from ``classification.json``, written by
``classify_rq2.py`` from text statistics **before** any timing. Reading them from a
file rather than deciding here is what keeps the claim non-circular: "high
speculatability workloads get more speedup" is only informative if "high" was
settled without looking at speedup.

Control variables, in the advisor's terms: the n-gram order never changes across
the workload contrast, and the dataset never changes across the efficiency
contrast. Datastore size, tokenizer, request count, target length and draft budget
are identical in every cell.

The budget sweep from the earlier run is gone. With ``beta = 0`` -- verifying n
positions is one forward pass, which is the accounting the advisor confirmed --
there is no penalty term on B, so ``L/S`` rises monotonically with it and there is
no optimum to find. The peak the earlier run reported came entirely from an assumed
``beta = 0.01``.

Why the run is cheap: every arm emits bit-identical drafts (asserted before any
timing, and a divergent cell aborts the run). Token accounting is therefore
identical across arms and is computed once per cell, while retrieval latency is
measured per arm under pinned, warmed-up conditions.

Phases per cell
---------------
1. **equivalence** -- all four arms must agree on every probe prompt
2. **accounting** -- playback over every request; yields ``L`` and ``S``
3. **latency** -- per-arm median/p95 ``generate_draft`` cost plus build cost
4. **cost model** -- speedup, excluding and including the amortized build

    python scripts/run_rq2.py --dry-run
    python scripts/run_rq2.py
    python scripts/run_rq2.py --datasets codesearch,xsum --store-sizes 200000
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402

from specdecode.costmodel import (  # noqa: E402
    DrafterCostProfile,
    TargetLatencyModel,
    WorkloadCounts,
    reuse_curve,
    speedup_both_ways,
)
from specdecode.experiments.equivalence import (  # noqa: E402
    DraftDivergence,
    assert_draft_equivalent,
    equivalence_prompts,
)
from specdecode.experiments.rq2_workloads import NotEnoughData, load_or_build  # noqa: E402
from specdecode.experiments.timing import (  # noqa: E402
    measure_timer_overhead_ns,
    pinned_threads,
    timed_build,
    warmup,
)
from specdecode.simulator.drafter.precomputeTensorNGramDrafter import (  # noqa: E402
    PrecomputeTensorNGramDrafter,
)
from specdecode.simulator.drafter.tensorNGramDrafter import (  # noqa: E402
    IndexedTensorNGramDrafter,
    NGramIndex,
    TensorNGramDrafter,
)
from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import (  # noqa: E402
    VectorizeTensorNGramDrafter,
)
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics  # noqa: E402
from specdecode.simulator.playback.speculativePlayback import (  # noqa: E402
    TensorSpeculativePlayback,
)
from specdecode.simulator.verifier.tensorGreedyVerifier import (  # noqa: E402
    TensorGreedyVerifier,
)

# Two independent implementations of the same contrast. Replicating the efficiency
# factor guards against the result being an artifact of one codebase rather than of
# the on-the-fly/precomputed distinction itself.
ARM_PAIRS = {
    "torch": ("torch_scan", "torch_index"),
    "python": ("python_scan", "python_index"),
}
ON_THE_FLY_ARMS = {"torch_scan", "python_scan"}

# beta = 0: verifying n draft positions is one forward pass, so extra positions are
# not charged. This is a definition of the unit of compute rather than a guess, and
# it removes the term the earlier budget sweep was reading its optimum off. A
# positive beta is reported as a sensitivity band, never as the headline.
DEFAULT_TARGET = TargetLatencyModel(
    t_verify_1_ns=20_000_000,   # 20 ms: a 7B-class model, one decode step
    beta=0.0,
    source="assumed",
    model_name="assumed-7b",
    hardware="assumed-gpu",
)
SENSITIVITY_BETAS = (0.005, 0.01)


def build_arms(
    corpus: Sequence[int], n: int, num_sequences: int, draft_depth: int
) -> Dict[str, object]:
    """One drafter per arm, plus the build cost each one paid.

    ``cap_positions`` is the corpus length so the index never truncates a gram's
    positions: truncation is lossless for depth drafting but not for width, and an
    arm that silently sees fewer branches is no longer draft-equivalent.
    """
    cap = len(corpus)
    arms: Dict[str, object] = {}
    builds: Dict[str, Dict[str, int]] = {}

    common = dict(n=n, num_sequences=num_sequences, draft_depth=draft_depth)

    arms["torch_scan"], rec = timed_build(
        "torch_scan", lambda: VectorizeTensorNGramDrafter(corpus_tokens=list(corpus), **common)
    )
    builds["torch_scan"] = {"ns": rec.elapsed_ns, "bytes": rec.approx_bytes}

    arms["torch_index"], rec = timed_build(
        "torch_index",
        lambda: PrecomputeTensorNGramDrafter(
            corpus_tokens=list(corpus), cap_positions=cap, **common
        ),
    )
    builds["torch_index"] = {"ns": rec.elapsed_ns, "bytes": rec.approx_bytes}

    arms["python_scan"], rec = timed_build(
        "python_scan", lambda: TensorNGramDrafter(corpus_tokens=list(corpus), **common)
    )
    builds["python_scan"] = {"ns": rec.elapsed_ns, "bytes": rec.approx_bytes}

    index, index_rec = timed_build(
        "ngram_index",
        lambda: NGramIndex(corpus_tokens=list(corpus), max_k=max(1, n - 1), cap_positions=cap),
    )
    arms["python_index"] = IndexedTensorNGramDrafter(index=index, **common)
    builds["python_index"] = {"ns": index_rec.elapsed_ns, "bytes": index_rec.approx_bytes}

    return {"arms": arms, "builds": builds}


def run_accounting(drafter: object, targets: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Token accounting over every request. Identical for all draft-equivalent arms."""
    verifier = TensorGreedyVerifier()
    total_L = 0
    total_S = 0
    accepted = 0
    rejected = 0
    per_request: List[float] = []

    for target in targets:
        metrics = PlaybackMetrics()
        playback = TensorSpeculativePlayback(None, drafter, verifier, metrics)  # type: ignore[arg-type]
        reconstructed = playback.run_tokens(list(target))
        if reconstructed != list(target):
            raise RuntimeError("playback did not reconstruct the target losslessly")
        total_L += metrics.normal_steps
        total_S += metrics.speculative_steps
        accepted += metrics.accepted_tokens
        rejected += metrics.rejected_tokens
        if metrics.speculative_steps:
            per_request.append(metrics.normal_steps / metrics.speculative_steps)

    return {
        "target_tokens": total_L,
        "speculative_steps": total_S,
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "acceptance_rate": accepted / max(accepted + rejected, 1),
        "token_speedup": total_L / max(total_S, 1),
        "token_speedup_median": statistics.median(per_request) if per_request else 1.0,
        "n_requests": len(targets),
    }


def measure_latency(
    drafter: object,
    prompts: Sequence[Sequence[int]],
    repeats: int,
    *,
    budget_s: float,
    min_calls: int,
) -> Dict[str, float]:
    """Median/p95 ``generate_draft`` cost under a pinned, warmed-up measurement.

    Sampling stops at whichever comes first: the requested repeats, or a wall-clock
    budget once ``min_calls`` samples exist. The arms differ by two orders of
    magnitude, so a fixed sample count would spend the whole run on the slowest arm
    while adding no precision -- a few hundred samples already pin a median.
    """
    samples: List[int] = []
    deadline = time.perf_counter() + budget_s
    for _ in range(repeats):
        for prompt in prompts:
            start = time.perf_counter_ns()
            drafter.generate_draft(list(prompt))  # type: ignore[attr-defined]
            samples.append(time.perf_counter_ns() - start)
        if len(samples) >= min_calls and time.perf_counter() > deadline:
            break
    samples.sort()
    return {
        "median_ns": float(statistics.median(samples)),
        "mean_ns": float(statistics.fmean(samples)),
        "p95_ns": float(samples[int(0.95 * (len(samples) - 1))]),
        "calls": len(samples),
    }


def sensitivity(counts: WorkloadCounts, profile: DrafterCostProfile) -> Dict[str, float]:
    """Speedup under positive beta, so the beta=0 choice is visible, not hidden."""
    out: Dict[str, float] = {}
    for beta in SENSITIVITY_BETAS:
        model = TargetLatencyModel(
            t_verify_1_ns=DEFAULT_TARGET.t_verify_1_ns, beta=beta,
            source="assumed", model_name=DEFAULT_TARGET.model_name,
            hardware=DEFAULT_TARGET.hardware,
        )
        out[f"beta_{beta}"] = speedup_both_ways(
            counts, profile, model
        )["excluding_build"].modeled_speedup
    return out


def cell_id(dataset: str, store: int, budget: int, seq: int) -> str:
    return f"{dataset}|{store}|B{budget}|S{seq}"


def load_classification(path: str) -> Dict[str, object]:
    """The pre-registered axis positions and pairs. Absent means stage 1 never ran."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            "Run scripts/classify_rq2.py first. The factorial must not decide which "
            "workloads are high-speculatability from its own speedups."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--datasets", default="",
        help="default: every dataset labelled low or high in classification.json",
    )
    parser.add_argument("--store-sizes", default="50000,200000,500000,1000000")
    parser.add_argument("--budget", type=int, default=4, help="draft depth; not swept")
    parser.add_argument("--num-sequences", type=int, default=1, help="1 = depth drafting")
    parser.add_argument("--n-gram", type=int, default=6, help="control variable; do not vary")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--latency-prompts", type=int, default=300)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--latency-budget-s", type=float, default=8.0)
    parser.add_argument("--latency-min-calls", type=int, default=200)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--classification",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "classification.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "results.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "cache"),
    )
    args = parser.parse_args()

    classification = load_classification(args.classification)
    axes: Dict[str, Dict[str, float]] = classification["axes"]  # type: ignore[assignment]
    pairs: List[Dict[str, object]] = classification.get("pairs", [])  # type: ignore[assignment]
    labels: Dict[str, str] = classification.get("labels", {})  # type: ignore[assignment]

    # Every measured dataset runs, not only the ones inside a pair: a dataset that
    # dominates nothing still carries a point on the speedup-vs-axis trend, which is
    # the evidence the 2x2 on its own cannot give.
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(axes)
    if not datasets:
        raise SystemExit("classification.json measured no dataset")

    stores = [int(s) for s in args.store_sizes.split(",") if s.strip()]
    cells = [(d, s) for d in datasets for s in stores]

    print(f"tokenizer      {args.tokenizer}")
    print(f"datasets       {datasets}")
    print(f"pairs          {len(pairs)}  " +
          ", ".join(f"{p['high']}>{p['low']}" for p in pairs[:6]))
    print(f"store sizes    {stores}")
    print(f"budget         B={args.budget}  num_sequences={args.num_sequences}  n={args.n_gram}")
    print(f"requests/cell  {args.n_requests} x {args.target_tokens} tokens")
    print(f"cells          {len(cells)}   arms/cell 4")
    print(f"beta           {DEFAULT_TARGET.beta}  (sensitivity at {SENSITIVITY_BETAS})")
    print(f"output         {args.output}\n")

    if args.dry_run:
        for d, s in cells:
            print(f"  {cell_id(d, s, args.budget, args.num_sequences)}   [{labels.get(d, '?')}]")
        print("\ndry run: nothing executed")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text)

    results: Dict[str, Dict] = {}
    if os.path.exists(args.output) and not args.force:
        with open(args.output, encoding="utf-8") as fh:
            results = json.load(fh).get("cells", {})
        print(f"resuming: {len(results)} cells already done\n")

    timer_overhead = measure_timer_overhead_ns()
    config = {
        "tokenizer": args.tokenizer,
        "n_gram": args.n_gram,
        "budget": args.budget,
        "num_sequences": args.num_sequences,
        "n_requests": args.n_requests,
        "target_tokens": args.target_tokens,
        "seed": args.seed,
        "timer_overhead_ns": timer_overhead,
        "target_latency_model": DEFAULT_TARGET.as_dict(),
        "sensitivity_betas": list(SENSITIVITY_BETAS),
        "arm_pairs": ARM_PAIRS,
        "axes": axes,
        "pairs": pairs,
        "labels": labels,
    }

    def save() -> None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"config": config, "cells": results}, fh, indent=2)

    started_all = time.time()
    for i, (dataset, store) in enumerate(cells, 1):
        cid = cell_id(dataset, store, args.budget, args.num_sequences)
        if cid in results and "error" not in results[cid] and not args.force:
            print(f"[{i}/{len(cells)}] skip {cid}", flush=True)
            continue

        print(f"[{i}/{len(cells)}] {cid}  [{labels.get(dataset, '?')}]", flush=True)
        t_cell = time.time()
        try:
            workload = load_or_build(
                args.cache_dir, dataset, enc, store,
                args.n_requests, args.target_tokens, sample_limit=args.sample_limit,
            )
            corpus, targets = workload.datastore, workload.targets

            with pinned_threads(1):
                built = build_arms(corpus, args.n_gram, args.num_sequences, args.budget)
                arms: Dict[str, object] = built["arms"]  # type: ignore[assignment]
                builds: Dict[str, Dict[str, int]] = built["builds"]  # type: ignore[assignment]

                rng = np.random.default_rng([args.seed, store, args.budget])
                probes = equivalence_prompts(
                    arms, max_prefix=args.n_gram - 1,
                    count=args.latency_prompts, rng=rng,
                )
                report = assert_draft_equivalent(arms, probes, context=cid)

                accounting = run_accounting(arms["torch_index"], targets)

                latency: Dict[str, Dict[str, float]] = {}
                for arm_id, drafter in arms.items():
                    warmup(drafter, probes[:64], rounds=2)
                    latency[arm_id] = measure_latency(
                        drafter, probes, args.latency_repeats,
                        budget_s=args.latency_budget_s,
                        min_calls=args.latency_min_calls,
                    )

            counts = WorkloadCounts(
                target_tokens=int(accounting["target_tokens"]),
                speculative_steps=int(accounting["speculative_steps"]),
                draft_budget_tokens=args.num_sequences * args.budget,
            )
            per_arm: Dict[str, Dict] = {}
            for arm_id, lat in latency.items():
                profile = DrafterCostProfile(
                    median_draft_ns=lat["median_ns"],
                    mean_draft_ns=lat["mean_ns"],
                    p95_draft_ns=lat["p95_ns"],
                    calls=int(lat["calls"]),
                    build_ns=builds[arm_id]["ns"],
                    build_bytes=builds[arm_id]["bytes"],
                    timer_overhead_ns=timer_overhead,
                )
                both = speedup_both_ways(
                    counts, profile, DEFAULT_TARGET,
                    amortization_requests=args.n_requests,
                )
                per_arm[arm_id] = {
                    "latency": lat,
                    "build_ns": builds[arm_id]["ns"],
                    "build_bytes": builds[arm_id]["bytes"],
                    "retrieval": "on_the_fly" if arm_id in ON_THE_FLY_ARMS else "precomputed",
                    "rho": both["excluding_build"].rho,
                    "drafter_cost_fraction": both["excluding_build"].drafter_cost_fraction,
                    "speedup_excluding_build": both["excluding_build"].modeled_speedup,
                    "speedup_including_build": both["including_build"].modeled_speedup,
                    "sensitivity": sensitivity(counts, profile),
                    "reuse_curve": reuse_curve(
                        counts, profile, DEFAULT_TARGET, [1, 10, 100, 1000, 10000]
                    ),
                }

            results[cid] = {
                "dataset": dataset,
                "label": labels.get(dataset, "?"),
                "axes": axes.get(dataset, {}),
                "store_tokens": store,
                "budget": args.budget,
                "num_sequences": args.num_sequences,
                "workload_shape": workload.as_dict(),
                "accounting": accounting,
                "arms": per_arm,
                "equivalence": {
                    "prompts_checked": report.prompts_checked,
                    "arms": list(report.arm_ids),
                },
                "seconds": time.time() - t_cell,
            }
            scan = per_arm["torch_scan"]["latency"]["median_ns"] / 1000
            idx = per_arm["torch_index"]["latency"]["median_ns"] / 1000
            print(
                f"        L/S={accounting['token_speedup']:.3f}"
                f"  accept={accounting['acceptance_rate']:.3f}"
                f"  scan={scan:.0f}us  index={idx:.0f}us"
                f"  ({time.time() - t_cell:.0f}s)",
                flush=True,
            )

        except DraftDivergence as exc:
            results[cid] = {"error": f"DraftDivergence: {exc}"}
            print(f"        ABORT: arms diverged\n{exc}", flush=True)
            save()
            raise
        except NotEnoughData as exc:
            results[cid] = {"error": f"NotEnoughData: {exc}"}
            print(f"        SKIPPED: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
            results[cid] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"        FAILED: {type(exc).__name__}: {exc}", flush=True)

        save()

    save()
    ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\ndone: {ok}/{len(results)} cells in {(time.time() - started_all)/60:.1f} min")
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
