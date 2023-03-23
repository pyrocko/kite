from .compound_sources import (
    CompoundModelProcessor,
    EllipsoidSource,
    PointCompoundSource,
)
from .okada import DislocProcessor, OkadaSource
from .pyrocko_gf import PyrockoMomentTensor, PyrockoProcessor, PyrockoRectangularSource

__sources__ = [
    OkadaSource,
    PyrockoMomentTensor,
    PyrockoRectangularSource,
    EllipsoidSource,
    PointCompoundSource,
]

__processors__ = [
    CompoundModelProcessor,
    PyrockoProcessor,
    DislocProcessor,
]
