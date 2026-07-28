"""
specdecode — a greedy speculative-decoding simulator.

Public API: the abstract contracts (interfaces), the concrete n-gram drafters /
greedy verifiers (both the list-based and the tensor-based depth/width variants),
the indexed drafter for large corpora, the playback loops, and the metrics tracker.
"""

from .interfaces import (
    AbstractDrafter,
    AbstractPlayback,
    AbstractTensorDrafter,
    AbstractTensorVerifier,
    AbstractVerifier,
)
from .simulator import (
    PAD_ID,
    GreedyVerifier,
    IndexedTensorNGramDrafter,
    NGramDrafter,
    NGramIndex,
    PlaybackMetrics,
    SpeculativePlayback,
    TensorGreedyVerifier,
    TensorNGramDrafter,
    TensorSpeculativePlayback,
)

__all__ = [
    # interfaces
    "AbstractDrafter",
    "AbstractVerifier",
    "AbstractPlayback",
    "AbstractTensorDrafter",
    "AbstractTensorVerifier",
    # drafters / verifiers
    "NGramDrafter",
    "GreedyVerifier",
    "TensorNGramDrafter",
    "TensorGreedyVerifier",
    "NGramIndex",
    "IndexedTensorNGramDrafter",
    # playback / metrics
    "PlaybackMetrics",
    "SpeculativePlayback",
    "TensorSpeculativePlayback",
    # sentinel
    "PAD_ID",
]
