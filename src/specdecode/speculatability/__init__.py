"""specdecode.speculatability — historical support and prefix branching entropy."""

from specdecode.speculatability.index import SuffixStatsIndex
from specdecode.speculatability.profile import (
    SpeculatabilityProfile,
    log_support,
    profile_fixed_k,
    profile_request,
    profile_with_backoff,
    quadrant,
)

__all__ = [
    "SpeculatabilityProfile",
    "SuffixStatsIndex",
    "log_support",
    "profile_fixed_k",
    "profile_request",
    "profile_with_backoff",
    "quadrant",
]
