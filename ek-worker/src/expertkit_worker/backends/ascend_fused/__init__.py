from .backend import AscendFusedBackend
from .op import AscendFusedComputeType
from .weights import (
    AscendFusedModelSlimW8A8WeightAdapter,
    AscendFusedWeightAdapter,
)

__all__ = [
    "AscendFusedBackend",
    "AscendFusedComputeType",
    "AscendFusedModelSlimW8A8WeightAdapter",
    "AscendFusedWeightAdapter",
]
