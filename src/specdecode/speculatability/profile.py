"""Per-request speculatability: how findable and how unambiguous the next token is.

Two measurements per target position, mirroring what the drafter actually does
(back off from the longest suffix until one has support):

* **support** -- occurrences of the matched suffix in the datastore. Zero support
  means the drafter has nothing to propose, however predictable the text is.
* **normalized branching entropy** -- how the continuations of that suffix are
  spread. Zero means the datastore names one continuation; one means it splits
  evenly and history tells you nothing about which branch to take.

They point in opposite directions -- high support is good, high entropy is bad --
so the combined score inverts entropy before multiplying.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence

from specdecode.speculatability.index import SuffixStatsIndex


@dataclass(frozen=True)
class SpeculatabilityProfile:
    """Summary of one (datastore, target) pair."""

    k_label: str  # "backoff" or "k=2"
    coverage: float  # fraction of positions whose suffix had support >= 1
    mean_support: float  # over covered positions
    median_support: float
    mean_branch_entropy: float  # bits, over covered positions
    normalized_branch_entropy: float  # in [0, 1], over covered positions
    mean_branch_factor: float
    oracle_accept_rate: float  # fraction where the top continuation is correct
    n_positions: int
    n_covered: int
    mean_matched_k: float  # average suffix length that actually matched

    @property
    def score(self) -> float:
        """coverage x (1 - normalized entropy), in [0, 1].

        Multiplicative, not additive: with zero support there is nothing to
        propose no matter how unambiguous the text would have been, and a
        perfectly ambiguous branch point is useless no matter how often it occurs.
        A sum would let one term paper over the other.
        """
        return self.coverage * (1.0 - self.normalized_branch_entropy)

    def as_dict(self) -> Dict[str, object]:
        return {
            "k_label": self.k_label,
            "coverage": self.coverage,
            "mean_support": self.mean_support,
            "median_support": self.median_support,
            "mean_branch_entropy": self.mean_branch_entropy,
            "normalized_branch_entropy": self.normalized_branch_entropy,
            "mean_branch_factor": self.mean_branch_factor,
            "oracle_accept_rate": self.oracle_accept_rate,
            "n_positions": self.n_positions,
            "n_covered": self.n_covered,
            "mean_matched_k": self.mean_matched_k,
            "score": self.score,
        }


def _summarize(
    k_label: str,
    supports: List[int],
    entropies: List[float],
    normalized: List[float],
    factors: List[int],
    oracle_hits: List[bool],
    matched_ks: List[int],
    n_positions: int,
) -> SpeculatabilityProfile:
    n_covered = len(supports)
    if n_covered == 0:
        return SpeculatabilityProfile(
            k_label=k_label, coverage=0.0, mean_support=0.0, median_support=0.0,
            mean_branch_entropy=0.0, normalized_branch_entropy=1.0,
            mean_branch_factor=0.0, oracle_accept_rate=0.0,
            n_positions=n_positions, n_covered=0, mean_matched_k=0.0,
        )

    return SpeculatabilityProfile(
        k_label=k_label,
        coverage=n_covered / n_positions if n_positions else 0.0,
        mean_support=sum(supports) / n_covered,
        median_support=float(median(supports)),
        mean_branch_entropy=sum(entropies) / n_covered,
        normalized_branch_entropy=sum(normalized) / n_covered,
        mean_branch_factor=sum(factors) / n_covered,
        # Denominator is all positions: an uncovered position is a guess the
        # drafter could not make, which is a miss, not an excluded sample.
        oracle_accept_rate=sum(oracle_hits) / n_positions if n_positions else 0.0,
        n_positions=n_positions,
        n_covered=n_covered,
        mean_matched_k=sum(matched_ks) / n_covered,
    )


def profile_with_backoff(
    index: SuffixStatsIndex,
    target_tokens: Sequence[int],
    *,
    max_k: Optional[int] = None,
) -> SpeculatabilityProfile:
    """Primary metric: at each position use the longest suffix that has support.

    This mirrors the drafter's backoff, so the measurement describes the workload
    the drafter actually faces rather than an idealized fixed-order matcher.
    """
    top_k = index.max_k if max_k is None else min(max_k, index.max_k)
    supports: List[int] = []
    entropies: List[float] = []
    normalized: List[float] = []
    factors: List[int] = []
    oracle_hits: List[bool] = []
    matched_ks: List[int] = []
    n_positions = 0

    for i in range(1, len(target_tokens)):
        n_positions += 1
        for k in range(min(top_k, i), 0, -1):
            gram = tuple(target_tokens[i - k : i])
            support = index.support(gram)
            if support == 0:
                continue
            supports.append(support)
            entropies.append(index.branch_entropy(gram))
            normalized.append(index.normalized_branch_entropy(gram))
            factors.append(index.branch_factor(gram))
            oracle_hits.append(index.top1_continuation(gram)[0] == target_tokens[i])
            matched_ks.append(k)
            break

    return _summarize(
        "backoff", supports, entropies, normalized, factors,
        oracle_hits, matched_ks, n_positions,
    )


def profile_fixed_k(
    index: SuffixStatsIndex, target_tokens: Sequence[int], k: int
) -> SpeculatabilityProfile:
    """Same measurement without backoff, for the per-order breakdown."""
    supports: List[int] = []
    entropies: List[float] = []
    normalized: List[float] = []
    factors: List[int] = []
    oracle_hits: List[bool] = []
    n_positions = 0

    for i in range(k, len(target_tokens)):
        n_positions += 1
        gram = tuple(target_tokens[i - k : i])
        support = index.support(gram)
        if support == 0:
            continue
        supports.append(support)
        entropies.append(index.branch_entropy(gram))
        normalized.append(index.normalized_branch_entropy(gram))
        factors.append(index.branch_factor(gram))
        oracle_hits.append(index.top1_continuation(gram)[0] == target_tokens[i])

    return _summarize(
        f"k={k}", supports, entropies, normalized, factors,
        oracle_hits, [k] * len(supports), n_positions,
    )


def profile_request(
    datastore_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    max_k: int = 3,
    ks: Sequence[int] = (1, 2, 3),
) -> Dict[str, SpeculatabilityProfile]:
    """Profile one request: the backoff metric plus a per-order breakdown."""
    index = SuffixStatsIndex(datastore_tokens, max_k=max_k)
    profiles = {"backoff": profile_with_backoff(index, target_tokens)}
    for k in ks:
        if k <= max_k:
            profiles[f"k={k}"] = profile_fixed_k(index, target_tokens, k)
    return profiles


def quadrant(
    support: float, entropy: float, support_split: float, entropy_split: float
) -> str:
    """Place a request on the support x entropy map, using data-driven splits.

    Axes point in opposite directions, so the diagonal of interest is Q4 (findable
    and unambiguous) against Q2 (neither) -- not Q1 against Q3.
    """
    high_support = support >= support_split
    low_entropy = entropy < entropy_split
    if high_support and low_entropy:
        return "Q4_high_spec"
    if not high_support and not low_entropy:
        return "Q2_low_spec"
    if high_support and not low_entropy:
        return "Q1_findable_but_ambiguous"
    return "Q3_unambiguous_but_rare"


def log_support(support: float) -> float:
    """Support on a log scale, which is how it is distributed in practice."""
    return math.log10(max(support, 1e-9))
