#!/bin/python
import warnings
import logging
import numpy as num

from pyrocko import guts
from kite.quadtree import QuadtreeConfig
from kite.meta import Subject, property_cached
logging.basicConfig(level=20)


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        return obj.__setattr__(variable, value)
    else:
        raise TypeError('value must be of type numpy.ndarray')


class UserIOWarning(UserWarning):
    pass


class SceneError(Exception):
    pass


class UTMFrameConfig(guts.Object):
    """
    Config object holding :py:class:`kite.scene.Scene` cobfiguration
    """
    zone = guts.String.T(default='None',
                         help='UTM zone of scene')
    llx = guts.Float.T(default=0.,
                       help='Scene UTM x-coordinate of lower left corner')
    lly = guts.Float.T(default=0.,
                       help='Scene UTM y-coordinate of lower left corner')
    dx = guts.Float.T(default=0.,
                      help='Scene pixel spacing in x direction (meter)')
    dy = guts.Float.T(default=0.,
                      help='Scene pixel spacing in y direction (meter)')


class UTMFrame(object):
    """UTM frame holding geographical references for
    :py:class:`kite.scene.Scene`
    """
    def __init__(self, scene, config=UTMFrameConfig()):
        self._scene = scene
        self.config = config
        self._scene.sceneChanged.subscribe(self._parseConfig)

        self._x = None
        self._y = None

    def _parseConfig(self):
        if (self.config.llx == 0. and self.config.lly == 0. and
                self.config.dx == 0. and self.config.dy == 0.) or\
                self._scene.displacement is None:
            return
        self._x = self.config.llx +\
            num.arange(self._scene.displacement.shape[0]) * self.config.dx
        self._y = self.config.lly +\
            num.arange(self._scene.displacement.shape[1]) * self.config.dy
        return

    def _updateExtent(self):
        if self.x is None or self.y is None:
            return
        llx, lly = self.x.min(), self.y.min()
        urx, ury = self.x.max(), self.y.max()

        self.config.llx = llx
        self.config.dx = abs(urx - llx)/self.x.size
        self.config.lly = lly
        self.config.dy = abs(ury - lly)/self.y.size
        self.config.regularize()

        self.grid_x = None
        self.grid_y = None
        self._extent = None
        return

    @property
    def x(self):
        """
        UTM x-vector, same size as ``Nx`` of ``Scene.displacement``.
        """
        return self._x

    @x.setter
    def x(self, value):
        if isinstance(value, float):
            value = num.repeat(value, 100)  # Fix this!
        self._x = num.sort(value)
        self._updateExtent()

    @property
    def y(self):
        """
        UTM y-vector, same size as ``xM`` of ``Scene.displacement``.
        """
        return self._y

    @y.setter
    def y(self, value):
        if isinstance(value, float):
            value = num.repeat(value, 100)
        self._y = num.sort(value)
        self._updateExtent()

    @property_cached
    def grid_x(self):
        """
        UTM grid holding x coordinates of all pixels in ``NxM`` matrix
        of ``Scene.displacement``.
        """
        valid_data = num.isnan(self._scene.displacement)
        grid_x = num.repeat(self.x[:, num.newaxis],
                            self._scene.displacement.shape[1],
                            axis=1)
        return num.ma.masked_array(grid_x, valid_data, fill_value=num.nan)

    @property_cached
    def grid_y(self):
        """
        UTM grid holding y coordinates of all pixels in ``NxM`` matrix
        of ``Scene.displacement``.
        """
        valid_data = num.isnan(self._scene.displacement)
        grid_y = num.repeat(self.y[num.newaxis, :],
                            self._scene.displacement.shape[0],
                            axis=0)
        return num.ma.masked_array(grid_y, valid_data, fill_value=num.nan)

    @property_cached
    def _extent(self):
        urx, ury = self.x.max(), self.y.max()

        return (self.config.llx, self.config.lly, urx, ury,
                self.config.dx, self.config.dy)

    def extent(self):
        """Get the UTM extent and pixel spacing of the LOS Displacement grid

        :returns: Corner coordinates and spatial deltas
        ll_x, ll_y, ur_x, ur_y, dx, dy
        :rtype: {tuple}
        """
        # Funny construction but we want to avoid unnecessary computation
        return self._extent

    def _mapGridXY(self, x, y):
        ll_x, ll_y, ur_x, ur_y, dx, dy = self.extent()
        return (ll_x + (x * dx),
                ll_y + (y * dy))


class Meta(guts.Object):
    """Meta configuration for ``Scene``.
    """
    scene_title = guts.String.T(default='Unnamed Scene')
    scene_id = guts.String.T(default='INSAR')
    scene_view = guts.String.T(default='ASCENDING')
    date_first_view = guts.Timestamp.T(default=0.0)
    date_second_view = guts.Timestamp.T(default=86400.0)
    satellite_name = guts.String.T(default='Unnamed Satellite')


class SceneConfig(guts.Object):
    """ Configuration object, gathering ``kite.Scene`` and
    sub-objects configuration.
    """
    meta = Meta.T(default=Meta(),
                  help='Scene metainformation.')
    utm = UTMFrameConfig.T(default=UTMFrameConfig(),
                           help='Scene UTMFrame configuration.')
    quadtree = QuadtreeConfig.T(default=QuadtreeConfig(),
                                help='Quadtree configuration.')


class Scene(object):
    """Scene holding satellite LOS ground dispacements measurements

    :param displacement: ``NxM`` matrix of displacement in LOS
    :type displacement: :py:class:`numpy.ndrray`
    :param theta: ``NxM`` matrix of theta towards LOS.
        Theta is look vector elevation angle towards satellite from horizon
        in radians - ``pi/2: up; -pi/2: down``
    :type theta: :py:class:`numpy.ndarray`
    :param phi: ``NxM`` matrix of phi towards LOS.
        Phi look vector orientation angle towards satellite in radians.
        (0: east, pi/2 north)
    :type phi: :py:class:`numpy.ndarray`
    :param meta: Meta information for the scene
    :type meta: :py:class:`kite.scene.Meta`

    :param los: Displacement measurements (displacement, theta, phi) from
        satellite measurements
    :type los: :py:class:`kite.scene.DisplacementLOS`
    """
    def __init__(self, config=SceneConfig()):
        self.config = config
        self.meta = self.config.meta
        self._log = logging.getLogger('Scene/%s' % self.meta.scene_title)
        self.sceneChanged = Subject()

        self._displacement = None
        self._phi = None
        self._theta = None
        self.los = LOSUnitVectors(scene=self)
        self.utm = UTMFrame(scene=self, config=self.config.utm)

    @property
    def displacement(self):
        """InSAR scene displacement ``NxM`` matrix of type
        :py:class:`numpy:ndarray`
        """
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self.sceneChanged._notify()

    @property
    def phi(self):
        """``NxM`` matrix of phi towards satellite' line of sight.
        Phi look vector orientation angle towards satellite in radians.
        (0: east, pi/2 north)
        """
        return self._phi

    @phi.setter
    def phi(self, value):
        if isinstance(value, float):
            self._phi = value
        else:
            _setDataNumpy(self, '_phi', value)
        self.sceneChanged._notify()

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
        """``NxM`` matrix of theta towards satellite' line of sight.
        Theta is look vector elevation angle towards satellite from horizon
        in radians - ``pi/2: up; -pi/2: down``
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        if isinstance(value, float):
            self._theta = value
        else:
            _setDataNumpy(self, '_theta', value)
        self.sceneChanged._notify()

    @theta.getter
    def theta(self):
        if isinstance(self._theta, float):
            _a = num.empty_like(self.displacement)
            _a.fill(self._theta)
            return _a
        else:
            return self._theta

    @property_cached
    def quadtree(self):
        """References the scene's :py:class:`kite.quadtree.Quadtree` instance.
        """
        from kite.quadtree import Quadtree
        return Quadtree(scene=self, config=self.config.quadtree)

    @property_cached
    def plot(self):
        from kite.plot2d import PlotDisplacement2D
        return PlotDisplacement2D(self)

    def spool(self):
        """Start the spool GUI :py:class:`kite.spool.Spool` to inspect
        the scene.
        """
        if self.displacement is None:
            raise SceneError('Can not display an empty scene.')
        from kite.spool import Spool
        return Spool(scene=self)

    @classmethod
    def import_file(cls, filename, **kwargs):
        """Import displacement data from foreign file format.
        Supported formats are `Matlab` and `Gamma` Remote Sensing Software.

        **Matlab**

        Variable naming conventions for variables in Matlab ``.mat`` file:

        ================== ====================
        Property           Matlab ``.mat`` name
        ================== ====================
        Scene.displacement ``ig_``
        Scene.phi          ``phi``
        Scene.theta        ``theta``
        Scene.utm.x        ``xx``
        Scene.utm.x        ``yy``
        ================== ====================

        **Gamma**

        Support for GAMMA Remote Sensing binary files
        A ``.par`` file is expected in the import folder

        :param filename: Filename of resource to import
        :type filename: str
        :param **kwargs: keyword arguments passed to import function
        :type **kwargs: dict
        :returns: Scene from filename
        :rtype: {:py:class:`kite.Scene`}
        :raises: TypeError
        """
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
        scene.utm.x = data['utm_x']
        scene.utm.y = data['utm_y']

        return scene

    def save(self, scene_name=None):
        """Save kite scene to kite file structure

        Saves the current scene meta information and UTM frame to a YAML
        (``.yml``) file. Numerical data (``Scene.displacement``,
        ``Scene.theta`` and ``Scene.phi``) are saved as binary files from
        :py:class:`numpy.ndarray`.
        :param scene_name: Filenames to save scene to, defaults to
        ``Scene.meta.scene_id + _ + Scene.meta.scene_view``
        :type scene_name: {str}, optional
        """
        filename = scene_name or '%s_%s' % (self.meta.scene_id,
                                            self.meta.scene_view)

        self._log.info('Saving scene to %s.[yml,dsp,tht,phi]' % filename)

        components = {
            'displacement': 'dsp',
            'theta': 'tht',
            'phi': 'phi',
        }

        self.config.dump(filename='%s.yml' % filename,
                         header='kite.Scene YAML Config\n'
                                'values of 9999.0 == NaN')
        for comp, ext in components.iteritems():
            num.save(file='%s.%s' % (filename, ext),
                     arr=self.__getattribute__(comp))

    @classmethod
    def load(cls, scene_name):
        """Load a kite scene from file ``scene_name.[yml,dsp,tht,phi]``
        structure

        :param scene_name: Filenames the scene data is saved under
        :type scene_name: {str}
        :returns: Scene object from data resources
        :rtype: {:py:class:`kite.Scene`}
        """

        success = False
        components = {
            'displacement': 'dsp',
            'theta': 'tht',
            'phi': 'phi',
        }

        try:
            scene = cls()
            scene.config = guts.load(filename='%s.yml' % scene_name)
            success = True
        except IOError:
            raise UserIOWarning('Could not load %s.yml' % scene_name)

        for comp, ext in components.iteritems():
            try:
                data = num.load('%s.%s.npy' % (scene_name, ext))
                scene.__setattr__(comp, data)
                success = True
            except IOError:
                warnings.warn('Could not load %s from %s.%s'
                              % (comp.title(), scene_name, ext),
                              UserIOWarning)
        if not success:
            raise IOError('Could not load kite scene container %s'
                          % scene_name)
        return scene

    def __str__(self):
        return self.config.__str__()


class LOSUnitVectors(object):
    """Decomposed Line Of Sight vectors (LOS) derived from
    ``Scene.displacement``.
    """
    def __init__(self, scene):
        self._scene = scene
        self._scene.sceneChanged.subscribe(self._flush_vectors)

    def _flush_vectors(self):
        self.unitE = None
        self.unitN = None
        self.unitU = None

    @property_cached
    def unitE(self):
        """Unit vector in East, ``NxM`` matrix like ``Scene.displacement``
        """
        return num.cos(self._scene.phi) * num.cos(self._scene.theta)

    @property_cached
    def unitN(self):
        """Unit vector in North, ``NxM`` matrix like ``Scene.displacement``
        """
        return num.sin(self._scene.phi) * num.cos(self._scene.theta)

    @property_cached
    def unitU(self):
        """Unit vector Up, ``NxM`` matrix like ``Scene.displacement``
        """
        return num.sin(self._scene.theta)

    @property_cached
    def degTheta(self):
        """LOS incident angle in degree, ``NxM`` matrix like ``Scene.theta``
        """
        return num.rad2deg(self._scene.theta)

    @property_cached
    def degPhi(self):
        """LOS incident angle in degree, ``NxM`` matrix like ``Scene.phi``
        """
        return num.rad2deg(self._scene.phi)


class SceneTest(Scene):
    """Test scenes for synthetic displacements """

    @classmethod
    def createGauss(cls, nx=500, ny=500, **kwargs):
        scene = cls()
        scene.meta.scene_title = 'Synthetic Displacement | Gaussian'
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._gaussAnomaly(scene.utm.x, scene.utm.y,
                                                 **kwargs)
        return scene

    @classmethod
    def createRandom(cls, nx=500, ny=500, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Displacement | Uniform Random'
        scene = cls._prepareSceneTest(scene, nx, ny)

        rand_state = num.random.RandomState(seed=1010)
        scene.displacement = (rand_state.rand(nx, ny)-.5)*2

        return scene

    @classmethod
    def createSine(cls, nx=500, ny=500, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Displacement | Sine'
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._sineAnomaly(scene.utm.x, scene.utm.y,
                                                **kwargs)
        return scene

    @staticmethod
    def _prepareSceneTest(scene, nx=500, ny=500):
        scene.utm.x = num.linspace(0, nx, nx)
        scene.utm.y = num.linspace(0, ny, ny)
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, nx), ny).reshape((nx, ny))
        scene.phi = num.rot90(scene.theta)
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


__all__ = ['Scene', 'SceneConfig']


if __name__ == '__main__':
    testScene = SceneTest.createGauss()
