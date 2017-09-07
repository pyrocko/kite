import numpy as num

from . import compound_engine as ce
from .base import SandboxSource, SourceProcessor
from pyrocko.guts import Float

d2r = num.pi / 180.
r2d = 180. / num.pi
km = 1e3


class EllipsoidSource(SandboxSource):

    __implements__ = 'CompoundModel'

    length_x = Float.T(
        help='Length of x semi-axis of ellisoid before rotation in [m]',
        default=1.)
    length_y = Float.T(
        help='Length of y semi-axis of ellisoid before rotation in [m]',
        default=1.)
    length_z = Float.T(
        help='Length of z semi-axis of ellisoid before rotation in [m]',
        default=1.)
    rotation_x = Float.T(
        help='Clockwise rotation of ellipsoid around x-axis in [deg]',
        default=0.)
    rotation_y = Float.T(
        help='Clockwise rotation of ellipsoid around y-axis in [deg]',
        default=0.)
    rotation_z = Float.T(
        help='Clockwise rotation of ellipsoid around z-axis in [deg]',
        default=0.)

    mu = Float.T(
        help='Shear modulus, 2. Lame constant in [GPa]',
        default=8.,)
    lamda = Float.T(
        help='Lame constant in [GPa]',
        default=8.)
    cavity_pressure = Float.T(
        help='Pressure on the cavity walls in [GPa]',
        default=.5)

    @property
    def volume(self):
        K = self.lamda + 2 * self.mu / 3  # Bulk Modulus
        V = 4./3 * num.pi * self.length_x * self.length_y * self.length_z
        V = (self.cavity_pressure * V) / K
        return V

    def ECMParameters(self):
        params = {
            'x0': self.easting,
            'y0': self.northing,
            'z0': self.depth,
            'rotx': self.rotation_x,
            'roty': self.rotation_y,
            'rotz': self.rotation_z,
            'ax': self.length_x,
            'ay': self.length_y,
            'az': self.length_z,
            'P': self.cavity_pressure * 1e9,
            'mu': self.mu * 1e9,
            'lamda': self.lamda * 1e9
        }
        return params


class PointCompoundSource(SandboxSource):

    __implements__ = 'CompoundModel'

    rotation_x = Float.T(
        help='Clockwise rotation of ellipsoid around x-axis in [deg]',
        default=0.)
    rotation_y = Float.T(
        help='Clockwise rotation of ellipsoid around y-axis in [deg]',
        default=0.)
    rotation_z = Float.T(
        help='Clockwise rotation of ellipsoid around z-axis in [deg]',
        default=0.)
    dVx = Float.T(
        help='Volume change in x-plane in [m3]',
        default=1.)
    dVy = Float.T(
        help='Volume change in y-plane in [m3]',
        default=1.)
    dVz = Float.T(
        help='Volume change in z-plane in [m3]',
        default=1.)
    nu = Float.T(
        help='Poisson\'s ratio, typically 0.25',
        default=0.25)

    @property
    def volume(self):
        # After Nikkhoo, M et al. 2017
        return (self.dVx + self.dVy + self.dVz) / 1.8

    def pointCDMParameters(self):
        params = {
            'x0': self.easting,
            'y0': self.northing,
            'z0': self.depth,
            'rotx': self.rotation_x,
            'roty': self.rotation_y,
            'rotz': self.rotation_z,
            'dVx': self.dVx**3,
            'dVy': self.dVy**3,
            'dVz': self.dVz**3,
            'nu': self.nu
        }
        return params


class CompoundModelProcessor(SourceProcessor):

    __implements__ = 'CompoundModel'

    def process(self, sources, coords, nthreads=0):
        result = {
            'processor_profile': dict(),
            'displacement.e': num.zeros((coords.shape[0])),
            'displacement.n': num.zeros((coords.shape[0])),
            'displacement.d': num.zeros((coords.shape[0])),
        }

        for src in sources:
            if isinstance(src, EllipsoidSource):
                res = ce.ECM(coords, **src.ECMParameters())
            elif isinstance(src, PointCompoundSource):
                res = ce.pointCDM(coords, **src.pointCDMParameters())
            else:
                raise AttributeError('Source of wrong type!')
            result['displacement.e'] += res[0]
            result['displacement.n'] += res[1]
            result['displacement.d'] += res[2]

        return result
