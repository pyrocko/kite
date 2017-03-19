import numpy as num

from kite import okada_ext
from pyrocko.guts import Object, Float, Int, List, String
from .scene import Scene


class OkadaSource(Object):
    easting = Float.T(
        help='Easting in [m] (center?)')
    northing = Float.T(
        help='Northing in [m]')
    depth = Float.T(
        help='Depth in [m]')
    width = Float.T(
        help='Width, downdip in [m]')
    length = Float.T(
        help='Length in [m]')
    strike = Float.T(
        help='Strike, clockwise from north in [deg]')
    dip = Float.T(
        help='Dip, down from horizontal in [deg]')
    rake = Float.T(
        help='Rake, clockwise in [deg]; 0 is left-lateral Strike-Slip')
    opening = Float.T(
        help='Opening of the plane in [m]')

    def disloc_source(self):
        return num.array()


class OkadaPlane(OkadaSource):
    nplanes_width = Int.T(
        help='Number of discrete :class:`OkadaSources`, downdip')
    nplanes_length = Int.T(
        help='Number of discrete :class:`OkadaSource`, along strike')

    def disloc_source(self):
        return num.array([os.disloc_source() for os in self.planes])


class OkadaTrack(Object):

    class OkadaPolyPlane(OkadaPlane):
        id = String.T()

    planes__ = List.T(
        OkadaPolyPlane.T,
        optional=True)

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._planes = []

    @property
    def planes(self):
        pass

    @planes.setter
    def planes(self, planes):
        self._planes = planes

    @property
    def nodes(self):
        pass

    def add_node(self, easting, northing):
        pass

    def insert_node(self, easting, northing, pos):
        pass

    def __len__(self):
        return len(self.planes)

    def disloc_source(self):
        return num.array([os.disloc_source() for os in self.planes])


class SyntheticScene(Scene):
    def __init__(self, scene, *args, **kwargs):
        Scene.__init__(self, *args, **kwargs)
        self.ref_scene = scene

    @property
    def residual(self):
        return self.ref_scene.displacement - self.displacement


class DislocEngine(object):

    def process(self, frame, okada_source, nthreads=0):
        disloc_source = okada_source.disloc_source()

        syn = SyntheticScene()
        self.log.info('Calculating Okada displacement for %d planes'
                      % disloc_source.shape[0])

        syn.displacement = okada_ext.disloc(disloc_source)
        syn.frame = frame.copy()
