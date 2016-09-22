#!/bin/python
import warnings
import logging
import numpy as num
from pyrocko import guts
from pyrocko.guts_array import Array
from kite.meta import Subject, property_cached


logging.basicConfig(level=20)


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        return obj.__setattr__(variable, value)
    else:
        raise TypeError('value must be of type numpy.ndarray')


class UserIOWarning(UserWarning):
    pass


class LOSUnitVectors(object):
    def __init__(self, scene):
        self._scene = scene
        self._scene.sceneChanged.subscribe(self._flush_vectors)

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


class UTMFrame(guts.Object):
    """UTM frame holding geographical references for :python:`kite.Scene`
    """
    zone = guts.String.T(default='None',
                         help='UTM zone of scene')
    x__ = Array.T(default=num.array([]),
                  serialize_as='base64', serialize_dtype=num.float64,
                  help='UTM coordinate vector along x-axis of scene grid')
    y__ = Array.T(default=num.array([]),
                  serialize_as='base64', serialize_dtype=num.float64,
                  help='UTM coordinate vector along y-axis of scene grid')

    def __init__(self, *args, **kwargs):
        guts.Object.__init__(self)
        self._scene = kwargs.pop('scene', None)
        self._x = kwargs.pop('x', None)
        self._y = kwargs.pop('y', None)

    def setScene(self, scene):
        self._scene = scene

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if isinstance(value, float):
            value = num.repeat(value, 100)
        self._x = num.sort(value)
        self.grid_x = None

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if isinstance(value, float):
            value = num.repeat(value, 100)
        self._y = num.sort(value)
        self.grid_y = None

    @property_cached
    def grid_x(self):
        valid_data = num.isnan(self._scene.displacement)
        grid_x = num.repeat(self.x[:, num.newaxis],
                            self._scene.displacement.shape[1],
                            axis=1)
        return num.ma.masked_array(grid_x, valid_data, fill_value=num.nan)

    @property_cached
    def grid_y(self):
        valid_data = num.isnan(self._scene.displacement)
        grid_y = num.repeat(self.y[num.newaxis, :],
                            self._scene.displacement.shape[0],
                            axis=0)
        return num.ma.masked_array(grid_y, valid_data, fill_value=num.nan)

    @property_cached
    def _extent(self):
        ll_x, ll_y = self.x.min(), self.y.min()
        ur_x, ur_y = self.x.max(), self.y.max()

        dx = abs(ur_x - ll_x)/self.x.size
        dy = abs(ur_y - ll_y)/self.y.size
        return ll_x, ll_y, ur_x, ur_y, dx, dy

    def extent(self):
        """Get the UTM extent and pixel spacing of the LOS Displacement grid

        :returns: ll_x, ll_y, ur_x, ur_y, dx, dy
        :rtype: {tuple}
        """
        # Funny construction but we want to avoid unnecessary computation
        return self._extent

    def _mapGridXY(self, x, y):
        ll_x, ll_y, ur_x, ur_y, dx, dy = self.extent()
        return (ll_x + (x * dx),
                ll_y + (y * dy))


class MetaSatellite(guts.Object):
    scene_title = guts.String.T(default='Unnamed Scene')
    scene_id = guts.String.T(default='INSAR')
    scene_view = guts.String.T(default='ASCENDING')
    date_first_view = guts.Timestamp.T(default=1475107200.0)
    date_second_view = guts.Timestamp.T(default=1474416000.0)
    satellite_name = guts.String.T(default='Unnamed Satellite')


class Scene(guts.Object):
    """Scene holding satellite LOS ground dispacements measurements

    :param displacement: NxM matrix of displacement in LOS
    :type displacement: :py:class:`numpy.ndrray`
    :param theta: NxM matrix of theta towards LOS.
        Theta is look vector elevation angle towards satellite from horizon
        in radians. (pi/2: up; -pi/2: down)
    :type theta: :py:class:`numpy.ndarray`
    :param phi: NxM matrix of phi towards LOS.
        Phi look vector orientation angle towards satellite in radians.
        (0: east, pi/2 north)
    :type phi: :py:class:`numpy.ndarray`
    :param utm_x: UTM latitudal reference vector for
        displacement, theta, phi ndarrays (N)
    :type utm_x: :py:class:`numpy.ndarray`
    :param utm_y: UTM longitudal reference vector for
        displacement, theta, phi ndarrays (N)
    :type utm_y: :py:class:`numpy.ndarray`
    :param X: Derived meshed utm_y
    :type X: :py:class:`numpy.ndarray`
    :param X: Derived meshed utm_x
    :type Y: :py:class:`numpy.ndarray`

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
    meta = MetaSatellite.T(optional=True)
    utm = UTMFrame.T(optional=True)

    def __init__(self, **kwargs):
        guts.Object.__init__(self)
        self.sceneChanged = Subject()

        self._displacement = None
        self._phi = None
        self._theta = None
        self.meta = kwargs.get('meta', MetaSatellite())
        self.utm = kwargs.get('utm', UTMFrame(scene=self))
        self.los = LOSUnitVectors(scene=self)

        if 'utm' in kwargs.keys():
            self.utm.setScene(self)

        self._log = logging.getLogger('Scene/%s' % self.meta.scene_title)

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self.sceneChanged._notify()

    @property
    def phi(self):
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
    def import_file(cls, filename, **kwargs):
        """Import displacement data from foreign file format.
        Supported formats are `Matlab` and `Gamma`

        Matlab
        ======
        Variable naming conventions in Matlab variables in `.mat` file
        ============ ==================
        Property     .mat name contains
        ============ ==================
        Displacement `ig_`
        Phi          `phi`
        Theta        `theta`
        UTM_X        `xx`
        UTM_Y        `yy`
        ============ ==================

        Gamma
        =====
        Support for GAMMA Remote Sensing binary files

        A `.par` file is expected in the import folder

        :param filename: Filename of resource to import
        :type filename: str
        :param **kwargs: keyword arguments passed to import function
        :type **kwargs: dict
        :returns: Scene from filename
        :rtype: {:python:`kite.Scene`}
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

    def dump(self, scene_name=None):
        """Dump kite scene to file structure named `scene_name`

        Saves the current scene meta information and UTM frame to a YAML
        (`.yml`) file. Numerical data (`Scene.displacement`,
        `Scene.theta` and `Scene.phi`) are saved as binary files from
        :python:`numpy.ndarray`.
        :param scene_name: Filenames to save scene to, defaults to
        `Scene.meta.scene_id + _ + Scene.meta.scene_view`
        :type scene_name: str, optional
        """
        filename = scene_name or '%s_%s' % (self.meta.scene_id,
                                            self.meta.scene_view)

        self._log.info('Saving scene to %s.[yml,dsp,tht,phi]' % filename)

        components = {
            'displacement': 'dsp',
            'theta': 'tht',
            'phi': 'phi',
        }

        guts.Object.dump(self,
                         filename='%s.yml' % filename,
                         header='Kite YAML Config')
        for comp, ext in components.iteritems():
            num.save(file='%s.%s' % (filename, ext),
                     arr=self.__getattribute__(comp))

    @staticmethod
    def load(scene_name):
        """Load a kite scene from `scene_name.[yml,dsp,tht,phi]` structure

        :param scene_name: Filenames the scene data is saved under
        :type scene_name: str
        :returns: Scene object from data resources
        :rtype: {:python:`kite.Scene`}
        """
        success = False
        components = {
            'displacement': 'dsp',
            'theta': 'tht',
            'phi': 'phi',
        }
        try:
            scene = guts.load(filename='%s.yml' % scene_name)
            success = True
        except IOError:
            scene = Scene()
            warnings.warn('Could not load %s.yml' % scene_name, UserIOWarning)

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


class SceneSynTest(Scene):
    """Test scenes for synthetic displacements """
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
