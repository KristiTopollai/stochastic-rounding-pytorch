"""SR-States public API."""

from .formats import StateFormat
from .optim._state import StateOptimizer
from .optim.adamw import AdamWReferenceRN, AdamWReferenceSR
from .optim.sgd import SGDMReferenceRN, SGDMReferenceSR
from .rounding import cast, register_rounding

__all__ = [
    "AdamWReferenceRN",
    "AdamWReferenceSR",
    "SGDMReferenceRN",
    "SGDMReferenceSR",
    "StateFormat",
    "StateOptimizer",
    "cast",
    "register_rounding",
]

__version__ = "0.1.0"
