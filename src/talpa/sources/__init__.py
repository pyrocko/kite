from .okada import OkadaSourceDelegate  # noqa
from .pyrocko import (PyrockoRectangularSourceDelegate,
                      PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate)


__sources__ = [OkadaSourceDelegate, PyrockoRectangularSourceDelegate,
               PyrockoMomentTensorDelegate, PyrockoDoubleCoupleDelegate]
