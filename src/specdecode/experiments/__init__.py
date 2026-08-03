"""specdecode.experiments — research harness for the RQ2 factorial study."""

from specdecode.experiments.equivalence import (
    DraftDivergence,
    EquivalenceReport,
    assert_draft_equivalent,
    diagnose_settings,
    equivalence_prompts,
    sample_prompts,
    saturated_prompts,
)
from specdecode.experiments.timing import (
    BuildRecord,
    estimate_bytes,
    measure_timer_overhead_ns,
    pinned_threads,
    timed_build,
    warmup,
)

__all__ = [
    "BuildRecord",
    "DraftDivergence",
    "EquivalenceReport",
    "assert_draft_equivalent",
    "diagnose_settings",
    "equivalence_prompts",
    "estimate_bytes",
    "measure_timer_overhead_ns",
    "pinned_threads",
    "sample_prompts",
    "saturated_prompts",
    "timed_build",
    "warmup",
]
