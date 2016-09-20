#!/bin/python
from pyrocko import guts
import numpy as num
import logging
from kite.meta import Subject, property_cached

SAR_META_KEYS = """
name
satellite
orbit
los
""".split()

logging.basicConfig(level=20)


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        return obj.__setattr__(variable, value)
    else:
        raise TypeError('value must be of type numpy.ndarray')


class Scene(Subject):
    """Scene holding satellite LOS ground dispacements measurements

    :param displacement: NxM matrix of displacement in LOS
    :type displacement: :py:class:`numpy.Array`
    :param theta: NxM matrix of theta towards LOS.
        Theta is look vector elevation angle towards satellite from horizon
        in radians. (pi/2: up; -pi/2: down)
    :type theta: :py:class:`numpy.Array`
    :param phi: NxM matrix of phi towards LOS.
        Phi look vector orientation angle towards satellite in radians.
        (0: east, pi/2 north)
    :type phi: :py:class:`numpy.Array`
    :param utm_x: UTM latitudal reference vector for
        displacement, theta, phi arrays (N)
    :type utm_x: :py:class:`numpy.Array`
    :param utm_y: UTM longitudal reference vector for
        displacement, theta, phi arrays (N)
    :type utm_y: :py:class:`numpy.Array`
    :param X: Derived meshed utm_y
    :type X: :py:class:`numpy.Array`
    :param X: Derived meshed utm_x
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
        # guts.Object.__init__(self)
        Subject.__init__(self)

        self.meta = MetaSatellite()
        self._log = logging.getLogger('Scene/%s' % self.meta.title)

        self._displacement = None
        self._phi = None
        self._theta = None
        self._utm_x = None
        self._utm_y = None
        # Meshed Grids
        self._utm_gridX = None
        self._utm_gridY = None

        self._quadtree = None

        # self.cartesian = DisplacementCartesian(self)
        self.los = LOSUnitVectors(self)

        self._log.debug('Instance created')

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self._notify()

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if isinstance(value, float):
            self._phi = value
        else:
            _setDataNumpy(self, '_phi', value)
        self._notify()

    @phi.getter
    def phi(self):
        if isinstance(self._phi, float):
            _a = num.empty_like(self.displacement)
            _a.fill(self._phi)
            return _a
        else:
            return self._phi

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if isinstance(value, float):
            self._theta = value
        else:
            _setDataNumpy(self, '_theta', value)
        self._notify()

    @theta.getter
    def theta(self):
        if isinstance(self._theta, float):
            _a = num.empty_like(self.displacement)
            _a.fill(self._theta)
            return _a
        else:
            return self._theta

    @property
    def utm_x(self):
        """Vector holding x-coordinates of the scene """
        return self._utm_x

    @utm_x.setter
    def utm_x(self, value):
        _setDataNumpy(self, '_utm_x', value)

    @property
    def utm_y(self):
        """Vector holding y-coordinates of the scene """
        return self._utm_y

    @utm_y.setter
    def utm_y(self, value):
        _setDataNumpy(self, '_utm_y', value)

    # Properties holding the meshed grids

    def _createMeshedGrids(self):
        _utm_gridX, _utm_gridY = num.meshgrid(self.utm_x, self.utm_y)
        nan = num.isnan(self.displacement)
        self._utm_gridX = num.ma.masked_array(_utm_gridX, nan)
        self._utm_gridY = num.ma.masked_array(_utm_gridY, nan)

    @property
    def utm_gridX(self):
        """Matrix holding meshed x-coordinates of the scene (read-only) """
        if self._utm_gridX is None:
            self._createMeshedGrids()
        return self._utm_gridX

    @property
    def utm_gridY(self):
        """Matrix holding meshed x-coordinates of the scene (read-only) """
        if self._utm_gridY is None:
            self._createMeshedGrids()
        return self._utm_gridY

    @property_cached
    def _UTMExtent(self):
        """Get the UTM extent and pixel spacing of the LOS Displacement grid

        :returns: ll_x, ll_y, ur_x, ur_y, dx, dy
        :rtype: {tuple}
        """
        ll_x, ll_y = self.utm_x.min(), self.utm_y.min()
        ur_x, ur_y = self.utm_x.max(), self.utm_y.max()

        dx = abs(ur_x - ll_x)/self.utm_x.size
        dy = abs(ur_y - ll_y)/self.utm_y.size
        return ll_x, ll_y, ur_x, ur_y, dx, dy

    def UTMExtent(self):
        return self._UTMExtent

    def _mapGridToUTM(self, x, y):
        ll_x, ll_y, ur_x, ur_y, dx, dy = self.UTMExtent()
        return (ll_x + (x * dx),
                ll_y + (y * dy))

    @property_cached
    def quadtree(self):
        from kite.quadtree import Quadtree
        return Quadtree(self)

    @property_cached
    def plot(self):
        from kite.plot2d import PlotDisplacement2D
        return PlotDisplacement2D(self)

    def spool(self):
        from kite.spool import Spool
        return Spool(scene=self)

    @classmethod
    def load(cls, filename, **kwargs):
        from kite import scene_io

        scene = cls()
        data = None

        for mod in scene_io.__all__:
            module = eval('scene_io.%s()' % mod)
            if module.validate(filename, **kwargs):
                data = module.read(filename, **kwargs)
                scene._log.debug('Recognized format %s for file %s' %
                                 (mod, filename))
                break
        if data is None:
            raise TypeError('Could not recognize format for %s' % filename)

        scene.theta = data['theta']
        scene.phi = data['phi']
        scene.displacement = data['displacement']
        scene.utm_x = data['utm_x']
        scene.utm_y = data['utm_y']

        return scene


class LOSUnitVectors(object):
    def __init__(self, scene):
        self._scene = scene
        self._scene.subscribe(self._flush_vectors)

    def _flush_vectors(self):
        self.unitE = None
        self.unitN = None
        self.unitU = None

    @property_cached
    def unitE(self):
        return num.cos(self._scene.phi) * num.cos(self._scene.theta)

    @property_cached
    def unitN(self):
        return num.sin(self._scene.phi) * num.cos(self._scene.theta)

    @property_cached
    def unitU(self):
        return num.sin(self._scene.theta)

    @property_cached
    def degTheta(self):
        return num.rad2deg(self._scene.theta)

    @property_cached
    def degPhi(self):
        return num.rad2deg(self._scene.phi)


class DisplacementCartesian(object):
    """Cartesian displacement derived from Line Of Sight (LOS) displacement vector

    :param dE: NxM matrix of displacement in East direction
    :type dE: :py:class:`numpy.Array`
    :param dN: NxM matrix of displacement in North direction
    :type dN: :py:class:`numpy.Array`
    :param dU: NxM matrix of displacement in Up direction
    :type dU: :py:class:`numpy.Array`
    """
    def _cached_displacement(component):
        """Factory method for cashed properties """
        def u_getter(instance):
            if instance.__dict__.get(component, None) is None:
                instance._init_vectors()
            return instance.__dict__[component]

        def u_setter(instance, value):
            instance.__dict__[component] = value

        def u_doc():
            return "Cartesian displacement in component %s" % component[1:]

        return property(u_getter, u_setter, doc=u_doc())

    dE = _cached_displacement('dE')
    dN = _cached_displacement('dN')
    dU = _cached_displacement('dU')
    dr = _cached_displacement('dr')

    def __init__(self, scene):
        self._scene = scene

        self.meta = self._scene.meta

        # self._flush_vectors()
        # self._scene.subscribe(self._flush_vectors)

    def _flush_vectors(self):
        self.dE = None
        self.dN = None
        self.dU = None
        self.dr = None

    def _init_vectors(self):
        """Initialise the cartesian vectors from LOS measurements """
        assert self._scene.displacement.shape \
            == self._scene.phi.shape \
            == self._scene.theta.shape, \
            'LOS displacement, phi, theta are not aligned.'

        self.dE = self._scene.displacement \
            * num.sin(self._scene.theta) * num.cos(self._scene.phi)
        self.dN = self._scene.displacement \
            * num.sin(self._scene.theta) * num.sin(self._scene.phi)
        self.dU = self._scene.displacement \
            * num.cos(self._scene.theta)
        self.dr = num.sqrt(self.dE**2 + self.dN**2 + self.dU**2) \
            * num.sign(self._scene.displacement)
        # self._dabs = self._dE + self._dN + self._dU

    @property
    def plot(self):
        if self.__dict__.get('_plot', None) is None:
            from kite.plot2d import PlotDisplacement
            self._plot = PlotDisplacement(self)
            self._plot.title = 'Displacement Cartesian'
            self._plot.default_component = 'dE'
        return self._plot


class Meta(guts.Object):
    title = guts.String.T(default='unnamed')
    satellite_name = guts.String.T(default='unnamed')


class MetaSatellite(Meta):
    pass


class SceneSynTest(Scene):
    """Test scene generating synthetic displacements """
    def __call__(self):
        return self.createGauss()

    @classmethod
    def createGauss(cls, nx=1000, ny=1000, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Input | Gaussian distribution'
        cls_dim = (nx, ny)

        scene.utm_x = num.linspace(2455, 3845, cls_dim[0])
        scene.utm_y = num.linspace(1045, 2403, cls_dim[1])
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, cls_dim[0]), cls_dim[1]) \
            .reshape(cls_dim)
        scene.phi = num.rot90(scene.theta)

        scene.displacement = scene._gaussAnomaly(scene.utm_x, scene.utm_y,
                                                 **kwargs)
        return scene

    @classmethod
    def createSine(cls, nx=1000, ny=1000, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Input | Sine distribution'
        cls_dim = (nx, ny)

        scene.utm_x = num.linspace(2455, 3845, cls_dim[0])
        scene.utm_y = num.linspace(1045, 2403, cls_dim[1])
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, cls_dim[0]), cls_dim[1]) \
            .reshape(cls_dim)
        scene.phi = num.rot90(scene.theta)

        scene.displacement = scene._sineAnomaly(scene.utm_x, scene.utm_y,
                                                **kwargs)
        return scene

    @staticmethod
    def _sineAnomaly(x, y, k1=.01, k2=.01, amplitude=3.):
        X, Y = num.meshgrid(x, y)
        return num.sin(k1 * X) * num.sin(k2 * Y)

    @staticmethod
    def _gaussAnomaly(x, y, sigma_x=.007, sigma_y=.005,
                      amplitude=3., x0=None, y0=None):
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
