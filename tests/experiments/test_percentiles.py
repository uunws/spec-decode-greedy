"""Tests for the robust latency statistics added to PlaybackMetrics."""

from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics, percentile_ns


def test_percentile_interpolates_between_samples() -> None:
    samples = [10, 20, 30, 40]
    assert percentile_ns(samples, 0.0) == 10.0
    assert percentile_ns(samples, 0.5) == 25.0
    assert percentile_ns(samples, 1.0) == 40.0


def test_percentile_handles_degenerate_inputs() -> None:
    assert percentile_ns([], 0.5) == 0.0
    assert percentile_ns([7], 0.95) == 7.0


def test_median_is_immune_to_a_single_outlier_but_the_mean_is_not() -> None:
    """The reason median is the headline: one OS preemption must not move it."""
    metrics = PlaybackMetrics()
    for _ in range(99):
        metrics.record_drafter_time(1_000)
    metrics.record_drafter_time(10_000_000)  # one scheduling hiccup

    assert metrics.median_drafter_wall_time_ns == 1_000
    assert metrics.average_drafter_wall_time_ns > 100_000  # mean is wrecked


def test_p95_exposes_a_heavy_tail_the_median_hides() -> None:
    metrics = PlaybackMetrics()
    for _ in range(90):
        metrics.record_drafter_time(1_000)
    for _ in range(10):
        metrics.record_drafter_time(500_000)

    assert metrics.median_drafter_wall_time_ns == 1_000
    assert metrics.p95_drafter_wall_time_ns > 100_000


def test_summary_exposes_median_and_p95_in_ms() -> None:
    metrics = PlaybackMetrics()
    metrics.record_drafter_time(1_000_000)
    metrics.record_drafter_time(3_000_000)

    summary = metrics.get_summary()

    assert summary["median_drafter_wall_time_ms"] == 2.0
    assert summary["p95_drafter_wall_time_ms"] == 2.9


def test_zero_calls_report_zero_rather_than_raising() -> None:
    metrics = PlaybackMetrics()
    assert metrics.median_drafter_wall_time_ns == 0.0
    assert metrics.p95_drafter_wall_time_ns == 0.0
