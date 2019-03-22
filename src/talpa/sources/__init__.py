from .okada import OkadaSourceDelegate
from .pyrocko import (PyrockoRectangularSourceDelegate,
                      PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate,
                      PyrockoRingfaultDelegate, PyrockoCLVDVolumeDelegate)
from .compound_models import (EllipsoidSourceDelegate,
                              PointCompoundSourceDelegate)


__sources__ = [OkadaSourceDelegate, PyrockoRectangularSourceDelegate,
               PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate,
               PyrockoRingfaultDelegate, PyrockoCLVDVolumeDelegate,
               EllipsoidSourceDelegate, PointCompoundSourceDelegate]
