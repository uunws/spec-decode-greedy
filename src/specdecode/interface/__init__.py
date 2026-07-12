"""
specdecode.interface — abstract contracts for drafter, verifier, and playback.

Public re-exports mirror the legacy ``specdecode.interfaces`` flat module so that
any code importing from this sub-package gets the same symbols.
"""

from specdecode.interface.abstractDrafter import AbstractDrafter
from specdecode.interface.abstractPlayback import AbstractPlayback
from specdecode.interface.abstractTensorDrafter import AbstractTensorDrafter
from specdecode.interface.abstractTensorVerifier import AbstractTensorVerifier
from specdecode.interface.abstractVerifier import AbstractVerifier

__all__ = [
    "AbstractDrafter",
    "AbstractVerifier",
    "AbstractPlayback",
    "AbstractTensorDrafter",
    "AbstractTensorVerifier",
]
