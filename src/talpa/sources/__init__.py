from .okada import OkadaSourceDelegate
from .pyrocko import (
    PyrockoRectangularSourceDelegate,
    PyrockoMomentTensorDelegate,
    PyrockoDoubleCoupleDelegate,
    PyrockoRingfaultDelegate,
    PyrockoVLVDSourceDelegate,
)
from .compound_models import EllipsoidSourceDelegate, PointCompoundSourceDelegate


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
