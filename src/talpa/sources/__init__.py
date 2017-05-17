from .okada import OkadaSourceDelegate
from .pyrocko import (PyrockoRectangularSourceDelegate,
                      PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate,
                      PyrockoRingfaultDelegate)
from .compound_models import (EllipsoidSourceDelegate,
                              PointCompoundSourceDelegate)


__sources__ = [OkadaSourceDelegate, PyrockoRectangularSourceDelegate,
               PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate,
               PyrockoRingfaultDelegate, EllipsoidSourceDelegate,
               PointCompoundSourceDelegate]
