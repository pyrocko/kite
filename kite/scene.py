#!/bin/python
from pyrocko import guts
import numpy as num
import logging
from kite.meta import Subject
from kite.plot2d import Plot2D

SAR_META_KEYS = """
name
satellite
orbit
los
""".split()

SAR_IO_MODULES = [
]

logging.basicConfig(level=20)


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        obj.__setattr__(variable, value)
    else:
        raise TypeError('value must be of type numpy.ndarray')


class Displacement(object):
    def __init__(self, scene):
        self._scene = scene

        self.meta = self._scene.meta
        self.plot = Plot2D(self)

    @property
    def x(self):
        return self._scene.x

    @property
    def y(self):
        return self._scene.y

    @property
    def X(self):
        return self._scene.X

    @property
    def Y(self):
        return self._scene.Y


class DisplacementCartesian(Displacement):
    """Cartesian displacement derived from Line Of Sight (LOS) displacement vector

    :param dx: NxM matrix of displacement in x direction
    :type dx: :py:class:`numpy.Array`
    :param dy: NxM matrix of displacement in y direction
    :type dy: :py:class:`numpy.Array`
    :param dz: NxM matrix of displacement in z direction
    :type dz: :py:class:`numpy.Array`
    """
    def __init__(self, scene):
        Displacement.__init__(self, scene)

        self._flush_vectors()
        self._scene.subscribe(self._flush_vectors)

        self.plot.title = 'Displacement Cartesian'
        self.plot.default_component = 'dx'

    def _flush_vectors(self):
        self._dx = None
        self._dy = None
        self._dz = None
        self._dr = None

    def _init_vectors(self):
        """Initialise the cartesian vectors from LOS measurements """
        assert self._scene.los.displacement.shape \
            == self._scene.los.phi.shape \
            == self._scene.los.theta.shape, \
            'LOS displacement, phi, theta are not aligned.'

        self._dx = self._scene.los.displacement \
            * num.sin(self._scene.los.theta) * num.cos(self._scene.los.phi)
        self._dy = self._scene.los.displacement \
            * num.sin(self._scene.los.theta) * num.sin(self._scene.los.phi)
        self._dz = self._scene.los.displacement \
            * num.cos(self._scene.los.theta)
        self._dr = num.sqrt(self._dx**2 + self._dy**2 + self._dz**2) \
            * num.sign(self._scene.los.displacement)
        # self._dabs = self._dx + self._dy + self._dz

    def _get_cached_property(self, prop):
        if self.__getattribute__(prop) is None:
            self._init_vectors()
        return self.__getattribute__(prop)

    @property
    def dx(self):
        return self._get_cached_property('_dx')

    @property
    def dy(self):
        return self._get_cached_property('_dy')

    @property
    def dz(self):
        return self._get_cached_property('_dz')

    @property
    def dr(self):
        return self._get_cached_property('_dr')


class DisplacementLOS(Displacement):
    """Displacement in Line Of Sight (LOS) from the satellite

    :param displacement: NxM matrix of displacement in LOS
    :type displacement: :py:class:`numpy.Array`
    :param theta: NxM matrix of theta towards LOS
    :type theta: :py:class:`numpy.Array`
    :param phi: NxM matrix of phi towards LOS
    :type phi: :py:class:`numpy.Array`
    :param x: Geographical reference pointing to :py:class:`kite.scene.Scene`
    :type x: :py:class:`numpy.Array`
    :param x: Geographical reference pointing to :py:class:`kite.scene.Scene`
    :type y: :py:class:`numpy.Array`
    :param X: Geographical reference pointing to :py:class:`kite.scene.Scene`
    :type X: :py:class:`numpy.Array`
    :param X: Geographical reference pointing to :py:class:`kite.scene.Scene`
    :type Y: :py:class:`numpy.Array`
     """
    def __init__(self, scene):
        Displacement.__init__(self, scene)

        self._displacement = None
        self._phi = None
        self._theta = None

        self.plot.title = 'Displacement LOS'
        self.plot.default_component = 'displacement'

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self._scene._notify()

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        _setDataNumpy(self, '_phi', value)
        self._scene._notify()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        _setDataNumpy(self, '_theta', value)
        self._scene._notify()


class Scene(guts.Object, Subject):
    """Scene holding satellite ground dispacements measurements

    :param x: Geographical latitudal reference for
                displacement, theta, phi arrays (N)
    :type x: :py:class:`numpy.Array`
    :param x: Geographical longitudal reference for
                displacement, theta, phi arrays (N)
    :type y: :py:class:`numpy.Array`
    :param X: Derived meshed latitudes
    :type X: :py:class:`numpy.Array`
    :param X: Derived meshed longitudes
    :type Y: :py:class:`numpy.Array`

    :param meta: Meta information for the scene
    :type meta: :py:class:`kite.scene.MetaSatellite`

    :param los: Displacement measurements (displacement, theta, phi) from
satellite measurements
    :type los: :py:class:`kite.scene.DisplacementLOS`

    :param cartesian: Derived cartesian displacements, derived from los
    :type cartesian: :py:class:`kite.scene.DisplacementCartesian`

    :param quadtree: Quadtree for the scene
    :type quadtree: :py:class:`kite.quadtree.Quadtree`
    """
    def __init__(self, **kwargs):
        """SARScene

        SARScene is initiated with spherical coordinates
        """
        guts.Object.__init__(self)
        Subject.__init__(self)

        self.meta = MetaSatellite()
        self.log = logging.getLogger('SARScene/%s' % self.meta.title)

        self._x = None
        self._y = None
        # Meshed Grids
        self._X = None
        self._Y = None
        # Placeholder
        self._quadtree = None

        self.cartesian = DisplacementCartesian(self)
        self.los = DisplacementLOS(self)

        self.log.debug('Instance created')

    @property
    def x(self):
        """Vector holding x-coordinates of the scene """
        return self._x

    @x.setter
    def x(self, value):
        _setDataNumpy(self, '_x', value)

    @property
    def y(self):
        """Vector holding y-coordinates of the scene """
        return self._y

    @y.setter
    def y(self, value):
        _setDataNumpy(self, '_y', value)

    # Properties holding the meshed grids

    def _createMeshedGrids(self):
        self._X, self._Y = num.meshgrid(self.x, self.y)

    @property
    def X(self):
        """Matrix holding meshed x-coordinates of the scene (read-only) """
        return self._X

    @X.getter
    def X(self):
        if self._X is None:
            self._createMeshedGrids()
        return self._X

    @property
    def Y(self):
        """Matrix holding meshed x-coordinates of the scene (read-only) """
        return self._Y

    @Y.getter
    def Y(self):
        if self._Y is None:
            self._createMeshedGrids()
        return self._Y

    @property
    def quadtree(self):
        return self._quadtree

    @quadtree.getter
    def quadtree(self):
        if self._quadtree is None:
            from kite.quadtree import Quadtree
            self._quadtree = Quadtree(self)
        return self._quadtree

    def mapRasterToCoordinates(self, x, y):
        return self.x[x], self.y[y]

    @classmethod
    def load(cls, filename, **kwargs):
        raise NotImplemented('Coming soon!')
        for module in SAR_IO_MODULES:
            if module.validate(filename, **kwargs):
                break
            raise TypeError('Could not recognize format for %s' % filename)
        module.read(filename, **kwargs)
        cls.los.theta = module.theta
        cls.los.phi = module.phi
        cls.los.displacement = module.displacement
        cls.los.x = module.x
        cls.los.y = module.y

        return cls


class SARIO(object):
    """ Prototype class for SARIO objects """

    def __init__(self):
        self.theta = None
        self.phi = None
        self.displacement = None
        self.x = None
        self.y = None

    def read(self, filename, **kwargs):
        """Read function of the file format

        :param filename: file to read
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        pass

    def write(self, filename, **kwargs):
        """Write method for IO

        :param filename: file to write to
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        pass

    def validate(self, filename, **kwargs):
        """Validate file format

        :param filename: file to validate
        :type filename: string
        :returns: Validation
        :rtype: {bool}
        """
        pass
        return False


class Meta(guts.Object):
    title = guts.String.T(default='unnamed')
    satellite_name = guts.String.T(default='unnanmed')
    # orbit = guts.String.T()


class MetaSatellite(Meta):
    pass


class SceneSynTest(Scene):
    """Test scene generating synthetic displacements """
    @classmethod
    def createGauss(cls, nx=1000, ny=1000, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Input - Gaussian distribution'
        cls_dim = (nx, ny)

        scene.x = num.linspace(2455, 3845, cls_dim[0])
        scene.y = num.linspace(1045, 2403, cls_dim[1])
        scene.los.theta = num.repeat(
            num.linspace(0.8, 0.85, cls_dim[0]), cls_dim[1]) \
            .reshape(cls_dim)
        scene.los.phi = num.rot90(scene.los.theta)

        scene.los.displacement = scene._gaussAnomaly(scene.x, scene.y,
                                                     **kwargs)
        return scene

    @staticmethod
    def _gaussAnomaly(x, y, sigma_x=.007, sigma_y=.005,
                      amplitude=3, x0=None, y0=None):
        if x0 is None:
            x0 = x.min() + abs(x.max()-x.min())/2
        if y0 is None:
            y0 = y.min() + abs(y.max()-y.min())/2
        X, Y = num.meshgrid(x, y)

        gauss_anomaly = amplitude * \
            num.exp(-(((X-x0)**2/2*sigma_x**2)+(Y-y0)**2/2*sigma_y**2))

        return gauss_anomaly

__all__ = """
Scene
""".split()


if __name__ == '__main__':
    testScene = SceneSynTest.createGauss()
