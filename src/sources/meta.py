import numpy as num

from ..meta import Subject
from pyrocko.guts import Object, Float

d2r = num.pi / 180.
r2d = 180. / num.pi
km = 1e3


class SandboxSource(Object):

    easting = Float.T(
        help='Easting in [m]')
    northing = Float.T(
        help='Northing in [m]')
    depth = Float.T(
        help='Depth in [m]')

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self.evParametersChanged = Subject()

    def parametersUpdated(self):
        self.evParametersChanged.notify()

    def getParametersArray(self):
        raise NotImplemented


class SandboxSourceRectangular(SandboxSource):

    width = Float.T(
        help='Width, downdip in [m]')
    length = Float.T(
        help='Length in [m]')
    strike = Float.T(
        default=45.,
        help='Strike, clockwise from north in [deg]; -180-180')
    dip = Float.T(
        default=45.,
        help='Dip, down from horizontal in [deg]; 0-90')
    rake = Float.T(
        default=90.,
        help='Rake, clockwise in [deg]; 0 is left-lateral Strike-Slip')
    slip = Float.T(
        default=1.5,
        help='Slip in [m]',
        optional=True)

    def outline(self):
        coords = num.empty((4, 2))

        c_strike = num.cos(self.strike * d2r)
        s_strike = num.sin(self.strike * d2r)
        c_dip = num.cos(self.dip * d2r)

        coords[0, 0] = s_strike * self.length/2
        coords[0, 1] = c_strike * self.length/2
        coords[1, 0] = -coords[0, 0]
        coords[1, 1] = -coords[0, 1]

        coords[2, 0] = coords[1, 0] - c_strike * c_dip * self.width
        coords[2, 1] = coords[1, 1] + s_strike * c_dip * self.width
        coords[3, 0] = coords[0, 0] - c_strike * c_dip * self.width
        coords[3, 1] = coords[0, 1] + s_strike * c_dip * self.width

        coords[:, 0] += self.easting
        coords[:, 1] += self.northing
        return coords

    @property
    def segments(self):
        yield self


class ProcessorProfile(dict):
    pass


class SourceProcessor(object):
    '''Interface definition of the processor '''
    __implements__ = 'ProcessorName'  # Defines What backend is implemented

    def __init__(self, model_scene):
        self._log = model_scene._log.getChild(self.__class__.__name__)

    def process(sources, coords, nthreads=0):
        raise NotImplementedError()

        result = {
            'displacement.n': num.array(),
            'displacement.e': num.array(),
            'displacement.d': num.array(),
            'processor_profile': ProcessorProfile()
        }
        return result
