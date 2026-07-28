"""
specdecode.simulator — concrete implementations package.

Re-exports all public symbols so that ``from specdecode.simulator import X``
continues to work.
"""

from specdecode.simulator.drafter.nGramDrafter import NGramDrafter
from specdecode.simulator.drafter.tensorNGramDrafter import (
    PAD_ID,
    IndexedTensorNGramDrafter,
    NGramIndex,
    TensorNGramDrafter,
)
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics
from specdecode.simulator.playback.speculativePlayback import (
    SpeculativePlayback,
    TensorSpeculativePlayback,
)
from specdecode.simulator.verifier.greedyVerifier import GreedyVerifier
from specdecode.simulator.verifier.tensorGreedyVerifier import TensorGreedyVerifier

__all__ = [
    "PAD_ID",
    "NGramDrafter",
    "GreedyVerifier",
    "TensorNGramDrafter",
    "TensorGreedyVerifier",
    "NGramIndex",
    "IndexedTensorNGramDrafter",
    "PlaybackMetrics",
    "SpeculativePlayback",
    "TensorSpeculativePlayback",
]
