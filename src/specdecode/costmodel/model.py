"""Turn the simulator's step accounting into a wall-clock speedup.

The simulator reports how many *steps* speculative decoding saved, which silently
assumes the drafter is free. That assumption erases the entire efficiency axis of
the RQ2 factorial: an on-the-fly scan and a precomputed lookup accept the same
tokens, so they produce identical step counts and identical speedups.

This module puts the cost back in::

    T_baseline = L * t_v(1)
    T_spec     = t_build / R + S * (t_d + t_v(B))
    speedup    = T_baseline / T_spec

where

* ``L`` -- target tokens (``metrics.normal_steps``)
* ``S`` -- speculative steps actually taken (``metrics.speculative_steps``)
* ``t_d`` -- measured drafter latency per call
* ``t_v(m)`` -- target-model latency when scoring ``m`` positions, from calibration
* ``B`` -- draft budget, ``num_sequences * draft_depth``
* ``R`` -- requests served by one datastore build

With ``t_v(m) = t_v(1) * (1 + beta * (m - 1))`` and the build excluded, this reduces
to the form the paper quotes::

    speedup = (L / S) / (1 + rho + beta * (B - 1)),   rho = t_d / t_v(1)

which is the whole argument in one line: ``L / S`` counts tokens per forward pass and
belongs to the data, while ``1 + rho + beta * (B - 1)`` prices one speculative step in
units of one baseline step and belongs to the system.

Note ``B`` rather than the draft *length*: a width draft asks the target model to
score every candidate position, so charging only the depth would invent an
advantage for width drafting that no real verifier enjoys.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

Provenance = Literal["calibrated", "assumed", "literature"]


@dataclass(frozen=True)
class TargetLatencyModel:
    """How long the target model takes for one verification forward pass.

    ``step_latency_ns(m) = t_verify_1_ns * (1 + beta * (m - 1))``: a fixed cost per
    forward pass plus a marginal cost for each *extra* position scored in the same
    pass. ``beta`` is small on real hardware -- verifying a few extra positions is
    nearly free while decoding is memory-bound -- which is what makes speculative
    decoding work at all.

    The ``m - 1`` matters. Scoring one position *is* ordinary decoding, so
    ``step_latency_ns(1)`` must return ``t_verify_1_ns`` exactly; the earlier
    ``beta * m`` charged a budget of 1 some 1% more than the baseline it is defined
    to equal. Ratios between cells were unaffected (the term cancels), but raw
    speedups were biased low and, more seriously, the penalty on ``B`` was
    overstated by a constant -- and the location of the optimum in a budget sweep is
    read directly off that penalty.
    """

    t_verify_1_ns: float
    beta: float
    source: Provenance
    model_name: str
    hardware: str
    fit_r2: Optional[float] = None
    max_calibrated_budget: Optional[int] = None

    def __post_init__(self) -> None:
        if self.t_verify_1_ns <= 0:
            raise ValueError("t_verify_1_ns must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")

    def step_latency_ns(self, budget_tokens: int) -> float:
        if budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")
        extra_positions = max(0, budget_tokens - 1)
        return self.t_verify_1_ns * (1.0 + self.beta * extra_positions)

    @property
    def is_measured(self) -> bool:
        return self.source == "calibrated"

    def as_dict(self) -> Dict[str, object]:
        return {
            "t_verify_1_ns": self.t_verify_1_ns,
            "beta": self.beta,
            "source": self.source,
            "model_name": self.model_name,
            "hardware": self.hardware,
            "fit_r2": self.fit_r2,
            "max_calibrated_budget": self.max_calibrated_budget,
        }


@dataclass(frozen=True)
class DrafterCostProfile:
    """Measured cost of one retrieval arm, from PlaybackMetrics + build timing."""

    median_draft_ns: float
    mean_draft_ns: float
    p95_draft_ns: float
    calls: int
    build_ns: int = 0
    build_bytes: int = 0
    timer_overhead_ns: float = 0.0

    def draft_ns(self, *, use_median: bool = True) -> float:
        """Per-call drafter latency, net of timer overhead.

        Median by default: the scanning arm has no early exit and a heavy tail, so
        its mean is dominated by outliers rather than by its typical behaviour.
        """
        raw = self.median_draft_ns if use_median else self.mean_draft_ns
        return max(0.0, raw - self.timer_overhead_ns)

    @classmethod
    def from_metrics(
        cls, metrics: object, *, timer_overhead_ns: float = 0.0
    ) -> "DrafterCostProfile":
        return cls(
            median_draft_ns=float(getattr(metrics, "median_drafter_wall_time_ns", 0.0)),
            mean_draft_ns=float(getattr(metrics, "average_drafter_wall_time_ns", 0.0)),
            p95_draft_ns=float(getattr(metrics, "p95_drafter_wall_time_ns", 0.0)),
            calls=int(getattr(metrics, "drafter_calls", 0)),
            build_ns=int(getattr(metrics, "build_wall_time_ns", 0)),
            build_bytes=int(getattr(metrics, "build_bytes", 0)),
            timer_overhead_ns=timer_overhead_ns,
        )


@dataclass(frozen=True)
class WorkloadCounts:
    """What the simulator measured, independent of any cost assumption."""

    target_tokens: int  # L
    speculative_steps: int  # S
    draft_budget_tokens: int  # B = num_sequences * draft_depth

    def __post_init__(self) -> None:
        if self.target_tokens < 0 or self.speculative_steps < 0:
            raise ValueError("token and step counts must be non-negative")
        if self.draft_budget_tokens < 0:
            raise ValueError("draft_budget_tokens must be non-negative")

    @property
    def token_accounting_speedup(self) -> float:
        """L / S -- the simulator's number, which assumes a free drafter."""
        if self.speculative_steps == 0:
            return 1.0
        return self.target_tokens / self.speculative_steps

    @classmethod
    def from_metrics(cls, metrics: object, *, draft_budget_tokens: int) -> "WorkloadCounts":
        return cls(
            target_tokens=int(getattr(metrics, "normal_steps", 0)),
            speculative_steps=int(getattr(metrics, "speculative_steps", 0)),
            draft_budget_tokens=draft_budget_tokens,
        )


@dataclass(frozen=True)
class ModeledSpeedup:
    """A speedup number together with everything needed to reproduce it."""

    token_accounting_speedup: float
    modeled_speedup: float
    rho: float
    amortized_build_ns_per_request: float
    drafter_cost_fraction: float
    ideal_speedup_zero_drafter_cost: float
    assumptions: Dict[str, object] = field(default_factory=dict)


def modeled_speedup(
    counts: WorkloadCounts,
    drafter: DrafterCostProfile,
    target: TargetLatencyModel,
    *,
    amortization_requests: int = 1,
    include_build: bool = True,
    use_median_draft: bool = True,
) -> ModeledSpeedup:
    """Speedup over vanilla decoding, charging the drafter for its time.

    ``include_build=False`` reports the steady state -- a datastore built offline
    and reused indefinitely, as in REST. ``include_build=True`` with
    ``amortization_requests=1`` reports the cold start -- a datastore built for a
    single request, as in Fast Context. Both are real operating points, so both
    are reported rather than one being chosen.
    """
    if amortization_requests < 1:
        raise ValueError("amortization_requests must be >= 1")

    t_v1 = target.step_latency_ns(1)
    t_v_budget = target.step_latency_ns(counts.draft_budget_tokens)
    t_d = drafter.draft_ns(use_median=use_median_draft)

    baseline_ns = counts.target_tokens * t_v1
    build_ns = (drafter.build_ns / amortization_requests) if include_build else 0.0
    spec_ns = build_ns + counts.speculative_steps * (t_d + t_v_budget)

    speedup = baseline_ns / spec_ns if spec_ns > 0 else 1.0
    ideal_ns = build_ns + counts.speculative_steps * t_v_budget
    ideal = baseline_ns / ideal_ns if ideal_ns > 0 else 1.0

    return ModeledSpeedup(
        token_accounting_speedup=counts.token_accounting_speedup,
        modeled_speedup=speedup,
        rho=t_d / t_v1,
        amortized_build_ns_per_request=build_ns,
        drafter_cost_fraction=t_d / (t_d + t_v_budget) if (t_d + t_v_budget) > 0 else 0.0,
        ideal_speedup_zero_drafter_cost=ideal,
        assumptions={
            "target": target.as_dict(),
            "draft_budget_tokens": counts.draft_budget_tokens,
            "amortization_requests": amortization_requests,
            "include_build": include_build,
            "draft_statistic": "median" if use_median_draft else "mean",
            "t_draft_ns": t_d,
            "t_verify_1_ns": t_v1,
            "t_verify_budget_ns": t_v_budget,
            "timer_overhead_ns": drafter.timer_overhead_ns,
        },
    )


def speedup_both_ways(
    counts: WorkloadCounts,
    drafter: DrafterCostProfile,
    target: TargetLatencyModel,
    *,
    amortization_requests: int = 1,
    use_median_draft: bool = True,
) -> Dict[str, ModeledSpeedup]:
    """Both operating points at once: datastore reused forever, and built per run."""
    common = dict(use_median_draft=use_median_draft)
    return {
        "excluding_build": modeled_speedup(
            counts, drafter, target, include_build=False, **common
        ),
        "including_build": modeled_speedup(
            counts,
            drafter,
            target,
            include_build=True,
            amortization_requests=amortization_requests,
            **common,
        ),
    }


def reuse_curve(
    counts: WorkloadCounts,
    drafter: DrafterCostProfile,
    target: TargetLatencyModel,
    reuse_counts: Sequence[int],
    *,
    use_median_draft: bool = True,
) -> List[Tuple[int, float]]:
    """Speedup as a function of how many requests share one datastore build.

    R=1 is the cold start and R -> infinity approaches the build-excluded number,
    so this curve has the two reported operating points as its endpoints.
    """
    return [
        (
            r,
            modeled_speedup(
                counts,
                drafter,
                target,
                amortization_requests=r,
                include_build=True,
                use_median_draft=use_median_draft,
            ).modeled_speedup,
        )
        for r in reuse_counts
    ]


def break_even_reuse(
    counts: WorkloadCounts,
    precomputed: DrafterCostProfile,
    on_the_fly: DrafterCostProfile,
    target: TargetLatencyModel,
    *,
    use_median_draft: bool = True,
) -> Optional[float]:
    """How many requests must share a build before precomputing wins.

    Returns ``None`` when precomputing never wins (its per-call latency is no
    better) or when it wins immediately at R=1.
    """
    t_d_pre = precomputed.draft_ns(use_median=use_median_draft)
    t_d_scan = on_the_fly.draft_ns(use_median=use_median_draft)
    per_call_saving = t_d_scan - t_d_pre
    if per_call_saving <= 0:
        return None

    total_saving = counts.speculative_steps * per_call_saving
    if total_saving <= 0:
        return None

    reuse = precomputed.build_ns / total_saving
    return None if reuse <= 1.0 else reuse


def break_even_rho(counts: WorkloadCounts, target: TargetLatencyModel) -> float:
    """The drafter cost, relative to one forward pass, at which speedup hits 1.0.

    Above this the drafter costs more than the steps it saves, and speculating is
    a net loss no matter how good the drafts are.
    """
    if counts.speculative_steps == 0:
        return math.inf
    t_v1 = target.step_latency_ns(1)
    t_v_budget = target.step_latency_ns(counts.draft_budget_tokens)
    slack_ns = counts.target_tokens * t_v1 - counts.speculative_steps * t_v_budget
    return max(0.0, slack_ns / (counts.speculative_steps * t_v1))
