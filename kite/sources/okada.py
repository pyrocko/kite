import numpy as np
from pyrocko import orthodrome as od
from pyrocko.guts import Bool, Float, List

from kite.sources import disloc_ext

from .base import SandboxSource, SandboxSourceRectangular, SourceProcessor

d2r = np.pi / 180.0
r2d = 180.0 / np.pi
km = 1e3


__all__ = ["OkadaSource", "OkadaPath", "DislocProcessor"]


class OkadaSource(SandboxSourceRectangular):
    """Rectangular Okada source model."""

    __implements__ = "disloc"

    opening = Float.T(help="Opening of the plane in [m]", optional=True, default=0.0)
    nu = Float.T(default=0.25, help="Poisson's ratio, typically 0.25")

    @property
    def seismic_moment(self):
        """Scalar Seismic moment

        Disregarding the opening (as for now)
        We assume a shear modulus of :math:`\mu = 36 \mathrm{GPa}`
        and :math:`M_0 = \mu A D`

        .. important ::

            We assume a perfect elastic solid with :math:`K=\\frac{5}{3}\\mu`

            Through :math:`\\mu = \\frac{3K(1-2\\nu)}{2(1+\\nu)}` this leads to
            :math:`\\mu = \\frac{8(1+\\nu)}{1-2\\nu}`

        :returns: Seismic moment release
        :rtype: float
        """
        mu = (8.0 * (1 + self.nu)) / (1 - 2.0 * self.nu)
        mu = 32e9  # GPa
        A = self.length * self.width
        return mu * A * self.slip

    @property
    def moment_magnitude(self):
        """Moment magnitude from Seismic moment

        We assume :math:`M_\\mathrm{w} = {\\frac{2}{3}}\\log_{10}(M_0) - 10.7`

        :returns: Moment magnitude
        :rtype: float
        """
        return 2.0 / 3 * np.log10(self.seismic_moment * 1e7) - 10.7

    def dislocSource(self, dsrc=None):
        if dsrc is None:
            dsrc = np.empty(10)

        dip = self.dip
        if self.dip == 90.0:
            dip -= 1e-2

        dsrc[0] = self.length
        dsrc[1] = self.width
        dsrc[2] = self.depth
        dsrc[3] = -dip
        dsrc[4] = self.strike - 180.0
        dsrc[5] = self.easting
        dsrc[6] = self.northing

        ss_slip = np.cos(self.rake * d2r) * self.slip
        ds_slip = np.sin(self.rake * d2r) * self.slip
        # print '{:<13}{}\n{:<13}{}'.format(
        #     'strike_slip', ss_slip, 'dip_slip', ds_slip)
        dsrc[7] = -ss_slip  # SS Strike-Slip
        dsrc[8] = -ds_slip  # DS Dip-Slip
        dsrc[9] = self.opening  # TS Tensional-Slip

        return dsrc

    # @property
    # def parameters(self):
    # return self.T.propnames

    def getParametersArray(self):
        return np.array([self.__getattribute__(p) for p in self.parameters])

    def setParametersArray(self, parameter_arr):
        if parameter_arr.size != len(self.parameters):
            raise AttributeError(
                "Invalid number of parameters, %s has %d" " parameters" % self.__name__,
                len(self.parameters),
            )
        for ip, param in enumerate(self.parameters):
            self.__setattr__(param, parameter_arr[ip])
        self.parametersUpdated()


class OkadaSegment(OkadaSource):
    enabled = Bool.T(default=True, optional=True)


class OkadaPath(SandboxSource):
    __implements__ = "disloc"

    depth = None
    nu = Float.T(default=0.25, help="Poisson's ratio, typically 0.25")
    nodes = List.T(
        default=[],
        optional=True,
        help="Nodes of the segments as (easting, northing) tuple of [m]",
    )
    segments__ = List.T(default=[], optional=True, help="List of all segments.")

    def __init__(self, *args, **kwargs):
        SandboxSource.__init__(self, *args, **kwargs)

        self._segments = []

        if not self.nodes:
            self.nodes.append([self.easting, self.northing])

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    @staticmethod
    def _newSegment(e1, n1, e2, n2, **kwargs):
        dE = e2 - e1
        dN = n2 - n1
        length = (dN**2 + dE**2) ** 0.5
        """Width Scaling relation after

        Leonard, M. (2010). Earthquake fault scaling: Relating rupture length,
            width, average displacement, and moment release, Bull. Seismol.
            Soc. Am. 100, no. 5, 1971-1988.
        """
        segment = {
            "northing": n1 + dN / 2,
            "easting": e1 + dE / 2,
            "depth": 0.0,
            "length": length,
            "width": 15.0 * length**0.66,
            "strike": np.arccos(dN / length) * r2d,
            "slip": 45.0,
            "rake": 90.0,
        }
        segment.update(kwargs)
        return OkadaSegment(**segment)

    def _moveSegment(self, pos, e1, n1, e2, n2):
        dE = e2 - e1
        dN = n2 - n1
        length = (dN**2 + dE**2) ** 0.5

        segment_update = {
            "northing": n1 + dN / 2,
            "easting": e1 + dE / 2,
            "length": length,
            "width": 15.0 * length**0.66,
            "strike": np.arccos(dN / length) * r2d,
        }

        segment = self.segments[pos]
        for attr, val in segment_update.items():
            segment.__setattr__(attr, val)

    def addNode(self, easting, northing):
        self.nodes.append([easting, northing])
        self.segments.append(
            self._newSegment(
                e1=self.nodes[-2][0],
                n1=self.nodes[-2][1],
                e2=self.nodes[-1][0],
                n2=self.nodes[-1][1],
            )
        )

    def insertNode(self, pos, easting, northing):
        self.nodes.insert(pos, [easting, northing])
        self.segments.append(
            self._newSegment(
                e1=self.nodes[pos][0],
                n1=self.nodes[pos][1],
                e2=self.nodes[pos + 1][0],
                n2=self.nodes[pos + 1][1],
            )
        )
        self._moveSegment(
            pos - 1,
            e1=self.nodes[pos - 1][0],
            n1=self.nodes[pos - 1][1],
            e2=self.nodes[pos][0],
            n2=self.nodes[pos][1],
        )

    def moveNode(self, pos, easting, northing):
        self.nodes[pos] = [easting, northing]
        if pos < len(self):
            self._moveSegment(
                pos,
                e1=self.nodes[pos][0],
                n1=self.nodes[pos][1],
                e2=self.nodes[pos + 1][0],
                n2=self.nodes[pos + 1][1],
            )
        if pos != 0:
            self._moveSegment(
                pos,
                e1=self.nodes[pos - 1][0],
                n1=self.nodes[pos - 1][1],
                e2=self.nodes[pos][0],
                n2=self.nodes[pos][1],
            )

    def __len__(self):
        return len(self.segments)

    def dislocSource(self):
        return np.array([seg.dislocSource() for seg in self.segments if seg.enabled])


class DislocProcessor(SourceProcessor):
    __implements__ = "disloc"

    @staticmethod
    def process(sources, sandbox, nthreads=0):
        result = {
            "processor_profile": dict(),
            "displacement.n": np.zeros(sandbox.frame.npixel),
            "displacement.e": np.zeros(sandbox.frame.npixel),
            "displacement.d": np.zeros(sandbox.frame.npixel),
        }

        src_nu = set(src.nu for src in sources)

        for nu in src_nu:
            nu_sources = [src for src in sources if src.nu == nu]
            nsources = len(nu_sources)
            src_arr = np.vstack([src.dislocSource() for src in nu_sources])

            north_shifts, east_shifts = od.latlon_to_ne_numpy(
                np.repeat(sandbox.frame.llLat, nsources),
                np.repeat(sandbox.frame.llLon, nsources),
                np.array([src.lat for src in nu_sources]),
                np.array([src.lon for src in nu_sources]),
            )

            src_arr[:, 5] += east_shifts
            src_arr[:, 6] += north_shifts

            res = disloc_ext.disloc(
                src_arr, sandbox.frame.coordinatesMeter, nu, nthreads
            )

            result["displacement.e"] += res[:, 0]
            result["displacement.n"] += res[:, 1]
            result["displacement.d"] += -res[:, 2]

        return result
