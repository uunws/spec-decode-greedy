"""Measurement helpers for the RQ2 harness.

Two things live here that the simulator cannot provide on its own:

* **Build cost.** Datastore construction happens outside the playback loop, so
  ``PlaybackMetrics`` never sees it. Left unmeasured, the precomputed arm appears
  to receive its index for free -- which is exactly the comparison RQ2 is making.
* **Measurement hygiene.** The drafters are fast enough that timer overhead and
  thread scheduling are a material fraction of what we are trying to measure.
"""

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class BuildRecord:
    """What one datastore construction cost."""

    label: str
    elapsed_ns: int
    approx_bytes: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "elapsed_ns": self.elapsed_ns,
            "approx_bytes": self.approx_bytes,
        }


def timed_build(
    label: str,
    factory: Callable[[], T],
    *,
    metrics: Optional[Any] = None,
    measure_bytes: bool = True,
) -> Tuple[T, BuildRecord]:
    """Construct an object, recording how long it took and how big it is.

    Passing ``metrics`` also forwards the record to ``record_build_time`` so the
    cost shows up in the run summary.
    """
    start_ns = time.perf_counter_ns()
    obj = factory()
    elapsed_ns = time.perf_counter_ns() - start_ns

    approx_bytes = estimate_bytes(obj) if measure_bytes else 0
    record = BuildRecord(label=label, elapsed_ns=elapsed_ns, approx_bytes=approx_bytes)

    if metrics is not None:
        metrics.record_build_time(elapsed_ns, label=label, approx_bytes=approx_bytes)
    return obj, record


def estimate_bytes(obj: Any) -> int:
    """Approximate retained size of a drafter or index.

    ``sys.getsizeof`` is shallow, and an ``NGramIndex`` is nested dicts of tuple
    keys and int lists, so its real footprint is orders of magnitude larger than
    the shallow number. This walks the structures the drafters actually use.
    """
    index = getattr(obj, "index", obj)
    tables = getattr(index, "tables", None)
    if tables is None:
        return _shallow_bytes(obj)

    total = sys.getsizeof(tables)
    for _, table in tables.items():
        total += sys.getsizeof(table)
        for key, positions in table.items():
            total += sys.getsizeof(key) + sys.getsizeof(positions)
    return total


def _shallow_bytes(obj: Any) -> int:
    tensor = getattr(obj, "corpus_tensor", None)
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.numel()
    try:
        return sys.getsizeof(obj)
    except TypeError:  # pragma: no cover - defensive
        return 0


def measure_timer_overhead_ns(samples: int = 10_000) -> float:
    """Cost of one ``perf_counter_ns()`` call, in nanoseconds.

    Worth recording in the results: the fastest arm is the one whose measured
    latency this overhead distorts most, and that is the arm we claim is faster.
    """
    if samples <= 0:
        return 0.0
    start = time.perf_counter_ns()
    for _ in range(samples):
        time.perf_counter_ns()
    return (time.perf_counter_ns() - start) / samples


@contextmanager
def pinned_threads(n: int = 1) -> Iterator[None]:
    """Pin torch to ``n`` threads for the duration of a measurement.

    Thread-count variation is a large source of run-to-run noise in the
    vectorized drafter, and it varies with machine load rather than with
    anything we are studying.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def warmup(drafter: Any, prompts: Sequence[Sequence[int]], rounds: int = 3) -> None:
    """Discard the first calls so cold caches do not land in the samples."""
    if not prompts:
        return
    for _ in range(rounds):
        for prompt in prompts:
            drafter.generate_draft(list(prompt))


def build_records_from(metrics: Any) -> List[BuildRecord]:
    """Re-read build records off a metrics object as typed records."""
    return [
        BuildRecord(
            label=str(record["label"]),
            elapsed_ns=int(record["elapsed_ns"]),
            approx_bytes=int(record["approx_bytes"]),
        )
        for record in getattr(metrics, "build_records", [])
    ]
