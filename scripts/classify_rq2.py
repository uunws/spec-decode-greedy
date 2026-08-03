"""Stage 1 of RQ2: place each candidate dataset on the two speculatability axes.

This runs **before** any timing and writes a file that the factorial then reads
without recomputing. That order is the whole point. If low/high were assigned after
seeing which dataset ran fast, "high-speculatability workloads are faster" would be
true by definition and would say nothing -- the circularity the advisor flagged.

What is measured, from text alone, with no model and no clock:

    History axis    mean_support                      (reported with coverage)
    Structure axis  1 - normalized_branch_entropy

Both come from ``profile_with_backoff``, which mirrors the drafter's own backoff, so
the numbers describe the workload the drafter actually faces.

Pairs, not labels, are what the factorial consumes. The first version of this script
split the candidates at the median of each axis and called a dataset high when it
was above both, but the two axes turn out to be **anti-correlated by construction**:
a suffix seen four hundred times almost certainly has many distinct continuations,
so high support drags structure down. Under a median split that leaves almost every
dataset straddling, and it would do so for any candidate set, not just this one.

So the rule is now **dominance**. A pair ``(high, low)`` is admissible when the high
member beats the low member on *both* axes at once. That needs no combined score --
which is the honest position, since no formula for one exists yet -- and it never
forces a dataset whose axes disagree into a bucket it does not belong in. Datasets
that dominate nothing and are dominated by nothing still get measured and still
appear in the trend figure; they simply carry no 2x2 cell.

Everything that could inflate an axis for free is held equal across datasets:

    same datastore size    a small corpus raises Structure on its own
    same tokenizer         fertility differences change what a token is
    same n-gram order      the control variable the design fixes
    same target count      and the same per-target token cap

    python scripts/classify_rq2.py --dry-run
    python scripts/classify_rq2.py
"""

import argparse
import json
import os
import sys
import time
from statistics import median
from typing import Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from specdecode.experiments.rq2_workloads import (  # noqa: E402
    CANDIDATES,
    NotEnoughData,
    load_or_build,
)
from specdecode.speculatability import SuffixStatsIndex, profile_with_backoff  # noqa: E402


def measure(
    datastore: Sequence[int], targets: Sequence[Sequence[int]], n: int
) -> Dict[str, float]:
    """Pool over targets before dividing, so long targets are not down-weighted.

    Averaging per-target means would give a 70-token request the same weight as a
    200-token one. Pooling the counts keeps the summary a property of the workload
    rather than of how its requests happen to be chopped up.
    """
    index = SuffixStatsIndex(list(datastore), max_k=max(1, n - 1))

    positions = 0
    covered = 0
    support_sum = 0.0
    entropy_sum = 0.0
    for target in targets:
        profile = profile_with_backoff(index, list(target))
        positions += profile.n_positions
        covered += profile.n_covered
        support_sum += profile.mean_support * profile.n_covered
        entropy_sum += profile.normalized_branch_entropy * profile.n_covered

    if covered == 0:
        return {
            "coverage": 0.0, "mean_support": 0.0,
            "normalized_branch_entropy": 1.0, "structure": 0.0,
            "n_positions": positions, "n_covered": 0,
        }

    entropy = entropy_sum / covered
    return {
        "coverage": covered / max(positions, 1),
        "mean_support": support_sum / covered,
        "normalized_branch_entropy": entropy,
        "structure": 1.0 - entropy,
        "n_positions": positions,
        "n_covered": covered,
    }


def dominates(a: Dict[str, float], b: Dict[str, float], *, margin: float = 0.0) -> bool:
    """``a`` is more speculatable than ``b`` on both axes at once.

    ``margin`` is a relative floor on each axis, so a pair separated by less than
    measurement wobble is not admitted. Both axes must clear it; clearing one and
    tying on the other is not dominance.
    """
    support_gain = (a["mean_support"] - b["mean_support"]) / max(b["mean_support"], 1e-9)
    structure_gain = (a["structure"] - b["structure"]) / max(b["structure"], 1e-9)
    return support_gain > margin and structure_gain > margin


def dominance_pairs(
    rows: Dict[str, Dict[str, float]], *, margin: float = 0.05
) -> List[Dict[str, object]]:
    """Every admissible (high, low) pair, ordered by how far apart the two sit."""
    pairs: List[Dict[str, object]] = []
    for high in rows:
        for low in rows:
            if high == low or not dominates(rows[high], rows[low], margin=margin):
                continue
            pairs.append({
                "high": high,
                "low": low,
                "support_ratio": rows[high]["mean_support"] / max(rows[low]["mean_support"], 1e-9),
                "structure_ratio": rows[high]["structure"] / max(rows[low]["structure"], 1e-9),
            })
    pairs.sort(
        key=lambda p: float(p["support_ratio"]) * float(p["structure_ratio"]), reverse=True
    )
    return pairs


def label(rows: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """A reading aid for the plane figure, never the thing the factorial pairs on.

    Kept because a scatter with eight unlabelled points is hard to talk about, but
    the median split it uses is exactly the rule that failed, so nothing downstream
    is allowed to depend on it.
    """
    s_split = median(r["mean_support"] for r in rows.values())
    t_split = median(r["structure"] for r in rows.values())

    labels: Dict[str, str] = {}
    for name, row in rows.items():
        high_support = row["mean_support"] >= s_split
        high_structure = row["structure"] >= t_split
        if high_support and high_structure:
            labels[name] = "high"
        elif not high_support and not high_structure:
            labels[name] = "low"
        else:
            labels[name] = "mixed"
    return labels


def report(
    rows: Dict[str, Dict[str, float]],
    labels: Dict[str, str],
    pairs: List[Dict[str, object]],
    failed: Dict[str, str],
    *,
    margin: float,
) -> None:
    """The whole of stage 1 on one screen, sorted by Structure."""
    print("\n" + "=" * 78)
    print(f"{'dataset':<13}{'support':>10}{'structure':>11}{'coverage':>10}"
          f"{'positions':>11}  label")
    print("-" * 78)
    for name in sorted(rows, key=lambda k: -rows[k]["structure"]):
        row = rows[name]
        print(
            f"{name:<13}{row['mean_support']:>10.2f}{row['structure']:>11.3f}"
            f"{row['coverage']:>10.3f}{row['n_positions']:>11,}  {labels[name]}"
        )
    print("=" * 78)

    print(f"\ndominance pairs (high beats low on BOTH axes by > {margin:.0%}):")
    for pair in pairs:
        print(f"  {pair['high']:<12} > {pair['low']:<12}"
              f"  support x{pair['support_ratio']:.2f}"
              f"  structure x{pair['structure_ratio']:.2f}")
    if not pairs:
        print("  none")
    if failed:
        print(f"\nexcluded before measurement: {sorted(failed)}")

    print(f"\npairs available for the factorial: {len(pairs)}")
    if len(pairs) < 4:
        print("WARNING: fewer than 4 pairs. One pair cannot separate speculatability "
              "from any other difference between two datasets.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--datasets", default=",".join(CANDIDATES))
    parser.add_argument("--store-tokens", type=int, default=200_000)
    parser.add_argument("--n-requests", type=int, default=100)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--n-gram", type=int, default=6,
                        help="control variable; fixed, never varied across the contrast")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="relative gap each axis must clear for dominance")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "classification.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "cache"),
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"tokenizer     {args.tokenizer}")
    print(f"datasets      {datasets}")
    print(f"datastore     {args.store_tokens:,} tokens (identical for every dataset)")
    print(f"requests      {args.n_requests} x <= {args.target_tokens} tokens")
    print(f"n-gram        {args.n_gram}   (control variable)")
    print(f"output        {args.output}\n")

    if args.dry_run:
        print("dry run: nothing executed")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text)

    rows: Dict[str, Dict[str, float]] = {}
    shapes: Dict[str, Dict[str, object]] = {}
    failed: Dict[str, str] = {}

    for i, dataset in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {dataset}", flush=True)
        started = time.time()
        try:
            workload = load_or_build(
                args.cache_dir, dataset, enc, args.store_tokens,
                args.n_requests, args.target_tokens, sample_limit=args.sample_limit,
            )
        except NotEnoughData as exc:
            failed[dataset] = str(exc)
            print(f"        EXCLUDED: {exc}", flush=True)
            continue

        row = measure(workload.datastore, workload.targets, args.n_gram)
        rows[dataset] = row
        shapes[dataset] = workload.as_dict()
        print(
            f"        support={row['mean_support']:.2f}"
            f"  structure={row['structure']:.3f}"
            f"  coverage={row['coverage']:.3f}"
            f"  ({time.time() - started:.0f}s)",
            flush=True,
        )

    if not rows:
        raise SystemExit("no dataset produced a measurement")

    labels = label(rows)
    pairs = dominance_pairs(rows, margin=args.margin)
    report(rows, labels, pairs, failed, margin=args.margin)

    payload = {
        "config": {
            "tokenizer": args.tokenizer,
            "store_tokens": args.store_tokens,
            "n_requests": args.n_requests,
            "target_tokens": args.target_tokens,
            "n_gram": args.n_gram,
            "pair_rule": f"dominance: high beats low on both axes by > {args.margin:.0%}",
            "margin": args.margin,
            "note": "pairs are relative to this candidate set and are not portable",
        },
        "axes": rows,
        "pairs": pairs,
        "labels": labels,
        "shapes": shapes,
        "excluded": failed,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwritten to {args.output}")
    print("This file is the pre-registration. Do not edit it after seeing speedups.")


if __name__ == "__main__":
    main()
