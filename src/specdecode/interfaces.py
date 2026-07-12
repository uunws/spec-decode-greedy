"""
Legacy shim — preserved for backward compatibility.

All abstract base classes have been moved to ``specdecode.interface.*``.
This module re-exports them so that existing imports of the form::

    from specdecode.interfaces import AbstractDrafter

continue to work unchanged.
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
