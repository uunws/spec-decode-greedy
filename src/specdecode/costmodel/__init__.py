"""specdecode.costmodel — converts step accounting into wall-clock speedup."""

from specdecode.costmodel.model import (
    DrafterCostProfile,
    ModeledSpeedup,
    TargetLatencyModel,
    WorkloadCounts,
    break_even_reuse,
    break_even_rho,
    modeled_speedup,
    reuse_curve,
    speedup_both_ways,
)

__all__ = [
    "DrafterCostProfile",
    "ModeledSpeedup",
    "TargetLatencyModel",
    "WorkloadCounts",
    "break_even_reuse",
    "break_even_rho",
    "modeled_speedup",
    "reuse_curve",
    "speedup_both_ways",
]
