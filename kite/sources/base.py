import numpy as num
from pyrocko import orthodrome as od
from pyrocko.guts import Float, Object

from kite.util import Subject

d2r = num.pi / 180.0
r2d = 180.0 / num.pi
km = 1e3


class SandboxSource(Object):

    lat = Float.T(default=0.0, help="Latitude in [deg]")
    lon = Float.T(default=0.0, help="Longitude in [deg]")
    easting = Float.T(default=0.0, help="Easting in [m]")
    northing = Float.T(default=0.0, help="Northing in [m]")
    depth = Float.T(default=1.0 * km, help="Depth in [m]")

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._cached_result = None
        self.evParametersChanged = Subject()
        self._sandbox = None

    def parametersUpdated(self):
        self._cached_result = None
        self.evParametersChanged.notify()

    def getSandboxOffset(self):
        if not self._sandbox or (self.lat == 0.0 and self.lon == 0.0):
            return 0.0, 0.0
        return od.latlon_to_ne_numpy(
            self._sandbox.frame.llLat, self._sandbox.frame.llLon, self.lat, self.lon
        )

    def getParametersArray(self):
        raise NotImplementedError


class SandboxSourceRectangular(SandboxSource):

    width = Float.T(
        help="Width, downdip in [m]",
        default=10000.0,
    )
    length = Float.T(
        help="Length in [m]",
        default=10000.0,
    )
    strike = Float.T(
        default=45.0, help="Strike, clockwise from North in [deg]; -180-180"
    )
    dip = Float.T(default=45.0, help="Dip, down from horizontal in [deg]; 0-90")
    rake = Float.T(
        default=90.0, help="Rake, clockwise in [deg]; 0 is right-lateral strike slip"
    )
    slip = Float.T(default=1.5, help="Slip in [m]", optional=True)

    def outline(self):
        coords = num.empty((4, 2))

        c_strike = num.cos(self.strike * d2r)
        s_strike = num.sin(self.strike * d2r)
        c_dip = num.cos(self.dip * d2r)

        coords[0, 0] = s_strike * self.length / 2
        coords[0, 1] = c_strike * self.length / 2
        coords[1, 0] = -coords[0, 0]
        coords[1, 1] = -coords[0, 1]

        coords[2, 0] = coords[1, 0] - c_strike * c_dip * self.width
        coords[2, 1] = coords[1, 1] + s_strike * c_dip * self.width
        coords[3, 0] = coords[0, 0] - c_strike * c_dip * self.width
        coords[3, 1] = coords[0, 1] + s_strike * c_dip * self.width

        north_shift, east_shift = self.getSandboxOffset()

        coords[:, 0] += self.easting + east_shift
        coords[:, 1] += self.northing + north_shift

        return coords

    @property
    def segments(self):
        yield self

    @classmethod
    def fromPyrockoSource(cls, source, store_dir=None, **kwargs):
        d = dict(
            lat=source.lat,
            lon=source.lon,
            northing=source.north_shift,
            easting=source.east_shift,
            depth=source.depth,
            width=source.width,
            length=source.length,
            strike=source.strike,
            dip=source.dip,
            rake=source.rake,
            slip=source.slip,
        )

        if hasattr(cls, "decimation_factor"):
            d["decimation_factor"] = source.decimation_factor

        if hasattr(cls, "store_dir"):
            d["store_dir"] = store_dir

        return cls(**d)


class ProcessorProfile(dict):
    pass


class SourceProcessor(object):
    """Interface definition of the processor"""

    __implements__ = "ProcessorName"  # Defines which backend is implemented

    def __init__(self, sandbox_scene):
        self._log = sandbox_scene._log.getChild(self.__class__.__name__)

    def process(sources, coords, nthreads=0):
        raise NotImplementedError()

        result = {
            "displacement.n": num.array(),
            "displacement.e": num.array(),
            "displacement.d": num.array(),
            "processor_profile": ProcessorProfile(),
        }
        return result
