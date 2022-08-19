from .compound_models import EllipsoidSourceDelegate, PointCompoundSourceDelegate
from .okada import OkadaSourceDelegate
from .pyrocko import (
    PyrockoDoubleCoupleDelegate,
    PyrockoMomentTensorDelegate,
    PyrockoRectangularSourceDelegate,
    PyrockoRingfaultDelegate,
    PyrockoVLVDSourceDelegate,
)

__sources__ = [
    OkadaSourceDelegate,
    PyrockoRectangularSourceDelegate,
    PyrockoMomentTensorDelegate,
    PyrockoDoubleCoupleDelegate,
    PyrockoRingfaultDelegate,
    PyrockoVLVDSourceDelegate,
    EllipsoidSourceDelegate,
    PointCompoundSourceDelegate,
]
