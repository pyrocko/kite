import numpy as num

from kite import disloc_ext
from pyrocko.guts import Object, Float, Int, List, String

d2r = num.pi / 180.


class OkadaSource(Object):

    __implements__ = 'disloc'

    easting = Float.T(
        help='Easting in [m]')
    northing = Float.T(
        help='Northing in [m]')
    depth = Float.T(
        help='Depth in [m]')
    width = Float.T(
        help='Width, downdip in [m]')
    length = Float.T(
        help='Length in [m]')
    strike = Float.T(
        default=45.,
        help='Strike, clockwise from north in [deg]')
    dip = Float.T(
        default=45.,
        help='Dip, down from horizontal in [deg]')
    rake = Float.T(
        default=90.,
        help='Rake, clockwise in [deg]; 0 is left-lateral Strike-Slip')
    slip = Float.T(
        default=1.5,
        help='Slip in [m]')
    nu = Float.T(
        default=.25,
        help='Material parameter Nu in P s^-1')
    opening = Float.T(
        help='Opening of the plane in [m]',
        optional=True,
        default=0.)

    def disloc_source(self, dsrc=None):
        if dsrc is None:
            dsrc = num.empty(10)

        if self.dip == 90.:
            dip = self.dip - 1e-5
        else:
            dip = self.dip

        dsrc[0] = self.length
        dsrc[1] = self.width
        dsrc[2] = self.depth
        dsrc[3] = -dip  # Dip
        dsrc[4] = self.strike
        dsrc[5] = self.easting
        dsrc[6] = self.northing

        ss_slip = num.cos(self.rake * d2r) * self.slip
        ds_slip = num.sin(self.rake * d2r) * self.slip
        print '{:<13}{}\n{:<13}{}'.format(
            'strike_slip', ss_slip, 'dip_slip', ds_slip)
        dsrc[7] = ss_slip  # SS Strike-Slip
        dsrc[8] = ds_slip  # DS Dip-Slip
        dsrc[9] = self.opening  # TS Tensional-Slip
        return dsrc

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


class OkadaPlane(OkadaSource):
    nplanes_width = Int.T(
        help='Number of discrete :class:`OkadaSources`, downdip')
    nplanes_length = Int.T(
        help='Number of discrete :class:`OkadaSource`, along strike')

    def disloc_source(self):
        return num.array([os.disloc_source() for os in self.planes])


class OkadaTrack(Object):

    __implements__ = 'disloc'

    class OkadaPolyPlane(OkadaPlane):
        id = String.T()

    planes__ = List.T(optional=True)

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


class DislocProcessor(object):
    __implements__ = 'disloc'

    @staticmethod
    def process(sources, coords, nthreads=1):
        result = {
            'processor_profile': dict()
        }

        src_arr = num.array([src.disloc_source() for src in sources])
        res = disloc_ext.disloc(src_arr, coords, src.nu, nthreads)

        result['north'] = res[:, 0]
        result['east'] = res[:, 1]
        result['down'] = res[:, 2]

        return result
