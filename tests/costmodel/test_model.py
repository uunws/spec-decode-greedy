"""Tests for the wall-clock cost model."""

import math

import pytest

from specdecode.costmodel import (
    DrafterCostProfile,
    TargetLatencyModel,
    WorkloadCounts,
    break_even_reuse,
    break_even_rho,
    modeled_speedup,
    reuse_curve,
    speedup_both_ways,
)

# 20 ms per forward pass, extra scored positions nearly free -- a 7B-class GPU.
TARGET = TargetLatencyModel(
    t_verify_1_ns=20_000_000,
    beta=0.01,
    source="assumed",
    model_name="test-7b",
    hardware="test-gpu",
)

# 100 target tokens produced in 40 speculative steps -> 2.5x by step accounting.
COUNTS = WorkloadCounts(target_tokens=100, speculative_steps=40, draft_budget_tokens=4)


def _profile(draft_ns: float, build_ns: int = 0) -> DrafterCostProfile:
    return DrafterCostProfile(
        median_draft_ns=draft_ns, mean_draft_ns=draft_ns,
        p95_draft_ns=draft_ns, calls=40, build_ns=build_ns,
    )


def test_free_drafter_reproduces_the_token_accounting_speedup() -> None:
    """The anchor: with no drafter cost and no budget penalty, speedup is L/S."""
    flat = TargetLatencyModel(
        t_verify_1_ns=20_000_000, beta=0.0, source="assumed",
        model_name="m", hardware="h",
    )
    result = modeled_speedup(COUNTS, _profile(0.0), flat, include_build=False)

    assert result.token_accounting_speedup == pytest.approx(2.5)
    assert result.modeled_speedup == pytest.approx(2.5)
    assert result.rho == pytest.approx(0.0)


def test_a_budget_of_one_costs_exactly_one_baseline_step() -> None:
    """Scoring a single position is ordinary decoding, so beta must not be charged.

    The earlier form ``t_v(1)*(1 + beta*m)`` made a budget of 1 cost 1% more than the
    baseline it is defined to equal. Ratios between cells survived that (the term
    cancels) but the budget sweep did not: the optimum in B is read straight off the
    beta penalty, so a constant offset in it moves the reported peak.
    """
    assert TARGET.step_latency_ns(1) == pytest.approx(TARGET.t_verify_1_ns)
    assert TARGET.step_latency_ns(0) == pytest.approx(TARGET.t_verify_1_ns)
    # Each position past the first, and only those, costs beta.
    assert TARGET.step_latency_ns(5) == pytest.approx(TARGET.t_verify_1_ns * 1.04)

    free = modeled_speedup(
        WorkloadCounts(target_tokens=100, speculative_steps=40, draft_budget_tokens=1),
        _profile(0.0), TARGET, include_build=False,
    )
    assert free.modeled_speedup == pytest.approx(free.token_accounting_speedup)


def test_a_costly_drafter_erodes_the_speedup() -> None:
    cheap = modeled_speedup(COUNTS, _profile(10_000), TARGET, include_build=False)
    costly = modeled_speedup(COUNTS, _profile(20_000_000), TARGET, include_build=False)

    assert cheap.modeled_speedup > costly.modeled_speedup
    # rho is relative to t_v(1), which is 20 ms exactly: a drafter as slow as one
    # forward pass has rho = 1.
    assert costly.rho == pytest.approx(1.0)
    assert costly.drafter_cost_fraction > 0.4


def test_budget_is_charged_on_the_whole_draft_not_just_its_depth() -> None:
    """Width drafting must not look free: the target scores every candidate."""
    narrow = WorkloadCounts(target_tokens=100, speculative_steps=40, draft_budget_tokens=4)
    wide = WorkloadCounts(target_tokens=100, speculative_steps=40, draft_budget_tokens=16)

    assert (
        modeled_speedup(wide, _profile(0.0), TARGET, include_build=False).modeled_speedup
        < modeled_speedup(narrow, _profile(0.0), TARGET, include_build=False).modeled_speedup
    )


def test_build_cost_is_charged_only_when_asked_and_shrinks_with_reuse() -> None:
    profile = _profile(10_000, build_ns=500_000_000)  # a 0.5 s index build

    excluded = modeled_speedup(COUNTS, profile, TARGET, include_build=False)
    cold = modeled_speedup(COUNTS, profile, TARGET, include_build=True,
                           amortization_requests=1)
    warm = modeled_speedup(COUNTS, profile, TARGET, include_build=True,
                           amortization_requests=1000)

    assert cold.modeled_speedup < warm.modeled_speedup < excluded.modeled_speedup
    assert cold.amortized_build_ns_per_request == 500_000_000
    assert warm.amortized_build_ns_per_request == 500_000


def test_speedup_both_ways_reports_the_two_operating_points() -> None:
    profile = _profile(10_000, build_ns=500_000_000)
    both = speedup_both_ways(COUNTS, profile, TARGET)

    assert set(both) == {"excluding_build", "including_build"}
    assert both["excluding_build"].modeled_speedup > both["including_build"].modeled_speedup


def test_reuse_curve_rises_towards_the_build_excluded_number() -> None:
    profile = _profile(10_000, build_ns=500_000_000)
    curve = reuse_curve(COUNTS, profile, TARGET, [1, 10, 100, 10_000])
    ceiling = modeled_speedup(COUNTS, profile, TARGET, include_build=False).modeled_speedup

    speedups = [s for _, s in curve]
    assert speedups == sorted(speedups)
    assert speedups[-1] < ceiling
    assert speedups[-1] == pytest.approx(ceiling, rel=0.01)


def test_break_even_reuse_answers_when_precomputing_pays_off() -> None:
    scanning = _profile(5_000_000)  # slow scan, no build
    precomputed = _profile(10_000, build_ns=500_000_000)

    reuse = break_even_reuse(COUNTS, precomputed, scanning, TARGET)

    assert reuse is not None
    assert reuse == pytest.approx(500_000_000 / (40 * (5_000_000 - 10_000)))


def test_break_even_reuse_is_none_when_precomputing_never_wins() -> None:
    scanning = _profile(10_000)
    precomputed = _profile(50_000, build_ns=500_000_000)  # slower per call too

    assert break_even_reuse(COUNTS, precomputed, scanning, TARGET) is None


def test_break_even_rho_marks_where_speculation_stops_paying() -> None:
    threshold = break_even_rho(COUNTS, TARGET)

    below = modeled_speedup(
        COUNTS, _profile(threshold * TARGET.t_verify_1_ns * 0.9), TARGET, include_build=False
    )
    above = modeled_speedup(
        COUNTS, _profile(threshold * TARGET.t_verify_1_ns * 1.1), TARGET, include_build=False
    )

    assert below.modeled_speedup > 1.0
    assert above.modeled_speedup < 1.0


def test_timer_overhead_is_subtracted_from_the_measured_latency() -> None:
    profile = DrafterCostProfile(
        median_draft_ns=1_000, mean_draft_ns=1_000, p95_draft_ns=1_000,
        calls=40, timer_overhead_ns=100,
    )
    assert profile.draft_ns() == 900


def test_median_and_mean_can_be_selected_explicitly() -> None:
    skewed = DrafterCostProfile(
        median_draft_ns=1_000, mean_draft_ns=900_000, p95_draft_ns=2_000_000, calls=40
    )
    by_median = modeled_speedup(COUNTS, skewed, TARGET, include_build=False)
    by_mean = modeled_speedup(
        COUNTS, skewed, TARGET, include_build=False, use_median_draft=False
    )

    assert by_median.modeled_speedup > by_mean.modeled_speedup


def test_assumptions_are_carried_with_every_number() -> None:
    result = modeled_speedup(COUNTS, _profile(10_000), TARGET, include_build=False)

    assert result.assumptions["target"]["source"] == "assumed"  # type: ignore[index]
    assert result.assumptions["draft_budget_tokens"] == 4
    assert result.assumptions["draft_statistic"] == "median"


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        TargetLatencyModel(t_verify_1_ns=0, beta=0.1, source="assumed",
                           model_name="m", hardware="h")
    with pytest.raises(ValueError):
        modeled_speedup(COUNTS, _profile(0.0), TARGET, amortization_requests=0)


def test_no_speculative_steps_yields_infinite_break_even_rho() -> None:
    empty = WorkloadCounts(target_tokens=0, speculative_steps=0, draft_budget_tokens=4)
    assert break_even_rho(empty, TARGET) == math.inf
    assert empty.token_accounting_speedup == 1.0
