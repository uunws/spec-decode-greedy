"""Tests for build-cost measurement and timing hygiene helpers."""

import time

import torch

from specdecode.experiments.timing import (
    build_records_from,
    estimate_bytes,
    measure_timer_overhead_ns,
    pinned_threads,
    timed_build,
    warmup,
)
from specdecode.simulator.drafter.precomputeTensorNGramDrafter import (
    PrecomputeTensorNGramDrafter,
)
from specdecode.simulator.drafter.tensorNGramDrafter import NGramIndex
from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import (
    VectorizeTensorNGramDrafter,
)
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics

CORPUS = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7, 8, 1, 2, 5, 9, 1, 2, 3]


def test_timed_build_records_elapsed_and_forwards_to_metrics() -> None:
    metrics = PlaybackMetrics()

    def factory() -> str:
        time.sleep(0.005)
        return "built"

    obj, record = timed_build("index", factory, metrics=metrics, measure_bytes=False)

    assert obj == "built"
    assert record.label == "index"
    assert record.elapsed_ns >= 5_000_000  # the 5 ms sleep is inside the window
    assert metrics.build_wall_time_ns == record.elapsed_ns
    assert len(metrics.build_records) == 1


def test_build_time_appears_in_summary() -> None:
    metrics = PlaybackMetrics()
    metrics.record_build_time(2_000_000, label="ngram_index", approx_bytes=4096)

    summary = metrics.get_summary()

    assert summary["build_calls"] == 1
    assert summary["build_wall_time_ms"] == 2.0
    assert summary["build_bytes"] == 4096


def test_precomputed_arm_costs_more_to_build_than_the_scanning_arm() -> None:
    """The asymmetry RQ2 depends on: only the precomputed arm pays a build cost."""
    _, scan = timed_build(
        "on_the_fly", lambda: VectorizeTensorNGramDrafter(corpus_tokens=CORPUS, n=3)
    )
    _, indexed = timed_build(
        "precomputed", lambda: PrecomputeTensorNGramDrafter(corpus_tokens=CORPUS, n=3)
    )

    assert indexed.elapsed_ns > 0
    assert indexed.approx_bytes > scan.approx_bytes


def test_estimate_bytes_walks_the_index_rather_than_measuring_it_shallowly() -> None:
    small = NGramIndex(corpus_tokens=CORPUS, max_k=1)
    large = NGramIndex(corpus_tokens=CORPUS * 20, max_k=2)

    assert estimate_bytes(large) > estimate_bytes(small)


def test_estimate_bytes_falls_back_to_the_corpus_tensor() -> None:
    drafter = VectorizeTensorNGramDrafter(corpus_tokens=CORPUS, n=3)
    assert estimate_bytes(drafter) == drafter.corpus_tensor.numel() * 8


def test_measure_timer_overhead_is_positive_and_small() -> None:
    overhead = measure_timer_overhead_ns(samples=2_000)
    assert 0.0 < overhead < 10_000.0  # nanoseconds; a microsecond would be alarming


def test_pinned_threads_restores_the_previous_setting() -> None:
    before = torch.get_num_threads()
    with pinned_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


def test_warmup_calls_the_drafter_without_recording_anything() -> None:
    metrics = PlaybackMetrics()
    drafter = VectorizeTensorNGramDrafter(corpus_tokens=CORPUS, n=3)

    warmup(drafter, [[1, 2], [2, 3]], rounds=2)

    assert metrics.drafter_calls == 0  # warmup must not pollute the samples


def test_build_records_round_trip_through_metrics() -> None:
    metrics = PlaybackMetrics()
    metrics.record_build_time(1_500, label="a", approx_bytes=10)
    metrics.record_build_time(2_500, label="b", approx_bytes=20)

    records = build_records_from(metrics)

    assert [r.label for r in records] == ["a", "b"]
    assert metrics.build_wall_time_ns == 4_000
    assert metrics.build_bytes == 30
