#!/bin/python
import warnings
import logging
import numpy as num
import utm

from pyrocko import guts
from kite.quadtree import QuadtreeConfig
from kite.meta import Subject, property_cached
from kite import scene_io
logging.basicConfig(level=20)


def greatCircleDistance(alat, alon, blat, blon):
    R1 = 6371009.
    d2r = num.deg2rad
    sin = num.sin
    cos = num.cos
    a = sin(d2r(alat-blat)/2)**2 + cos(d2r(alat)) * cos(d2r(blat))\
        * sin(d2r(alon-blon)/2)**2
    c = 2. * num.arctan2(num.sqrt(a), num.sqrt(1.-a))
    return R1 * c


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        return obj.__setattr__(variable, value)
    else:
        raise TypeError('value must be of type numpy.ndarray')


class UserIOWarning(UserWarning):
    pass


class SceneError(Exception):
    pass


class FrameConfig(guts.Object):
    """
    Config object holding :py:class:`kite.scene.Scene` cobfiguration
    """
    llLat = guts.Float.T(default=0.,
                         help='Scene latitude of lower left corner')
    llLon = guts.Float.T(default=0.,
                         help='Scene longitude of lower left corner')
    dLat = guts.Float.T(default=1.e-3,
                        help='Scene pixel spacing in x direction (degree)')
    dLon = guts.Float.T(default=1.e-3,
                        help='Scene pixel spacing in y direction (degree)')


class Frame(object):
    """UTM frame holding geographical references for
    :py:class:`kite.scene.Scene`
    """
    def __init__(self, scene, config=FrameConfig()):
        self._scene = scene
        self.config = config
        self._scene.sceneChanged.subscribe(self._updateExtent)

        self.extentE = 0.
        self.extentN = 0.
        self.spherical_distortion = 0.
        self.urE = 0.
        self.urN = 0.
        self.dN = 0.
        self.dE = 0.
        self.llN = None
        self.llE = None
        self.N = None
        self.E = None

    def _updateExtent(self):
        if self._scene.cols == 0 or self._scene.rows == 0:
            return
        self.llEutm, self.llNutm, self.utm_zone, self.utm_zone_letter =\
            utm.from_latlon(self.llLat, self.llLon)

        self.cols = self._scene.cols
        self.rows = self._scene.rows

        urlat = self.llLat + self.dLat * self.rows
        urlon = self.llLon + self.dLon * self.cols
        self.urEutm, self.urNutm, _, _ = utm.from_latlon(urlat, urlon,
                                                         self.utm_zone)

        # Width at the bottom of the scene
        self.extentE = greatCircleDistance(self.llLat, self.llLon,
                                           self.llLat, urlon)
        self.extentN = greatCircleDistance(self.llLat, self.llLon,
                                           urlat, self.llLon)

        # Width at the N' top of the scene
        extentE_top = greatCircleDistance(urlat, self.llLon,
                                          urlat, urlon)
        self.spherical_distortion = num.abs(self.extentE - extentE_top)

        self.dE = (self.extentE + extentE_top) / (2*self.cols)
        self.dN = self.extentN / self.rows
        self.E = num.arange(self.cols) * self.dE
        self.N = num.arange(self.rows) * self.dN

        self.llE = 0
        self.llN = 0
        self.urE = self.E.max()
        self.urN = self.N.max()

        self.config.regularize()

        self.gridE = None
        self.gridN = None
        return

    @property
    def llLat(self):
        return self.config.llLat

    @llLat.setter
    def llLat(self, llLat):
        self.config.llLat = llLat
        self._llLat = llLat
        self._updateExtent()

    @property
    def llLon(self):
        return self.config.llLon

    @llLon.setter
    def llLon(self, llLon):
        self.config.llLon = llLon
        self._llLon = llLon
        self._updateExtent()

    @property
    def dLat(self):
        return self.config.dLat

    @dLat.setter
    def dLat(self, dLat):
        self.config.dLat = dLat
        self._updateExtent()

    @property
    def dLon(self):
        return self.config.dLon

    @dLon.setter
    def dLon(self, dLon):
        self.config.dLon = dLon
        self._updateExtent()

    @property_cached
    def gridE(self):
        """
        UTM grid holding y coordinates of all pixels in ``NxM`` matrix
        of ``Scene.displacement``.
        """
        valid_data = num.isnan(self._scene.displacement)
        gridE = num.repeat(self.E[num.newaxis, :],
                           self.rows, axis=0)
        return num.ma.masked_array(gridE, valid_data, fill_value=num.nan)

    @property_cached
    def gridN(self):
        """
        UTM grid holding x coordinates of all pixels in ``NxM`` matrix
        of ``Scene.displacement``.
        """
        valid_data = num.isnan(self._scene.displacement)
        gridN = num.repeat(self.N[:, num.newaxis],
                           self.cols, axis=1)
        return num.ma.masked_array(gridN, valid_data, fill_value=num.nan)

    def _mapGridXY(self, x, y):
        ll_x, ll_y, ur_x, ur_y, dx, dy = self.extent()
        return (ll_x + (x * dx),
                ll_y + (y * dy))

    def __str__(self):
        return (
            'Lower right latitude:  {frame.llLat:.4f} N\n'
            'Lower right longitude: {frame.llLon:.4f} E\n'
            '\n\n'
            'UTM Zone:              {frame.utm_zone}{frame.utm_zone_letter}'
            'Lower right easting:   {frame.llE:.4f} m\n'
            'Lower right northing:  {frame.llN:.4f} m'
            '\n\n'
            'Pixel spacing east:    {frame.dE:.4f} m\n'
            'Pixel spacing north:   {frame.dN:.4f} m\n'
            'Extent east:           {frame.extentE:.4f} m\n'
            'Extent north:          {frame.extentN:.4f} m\n'
            'Dimensions:            {frame.cols} x {frame.rows} px\n'
            'Spherical distortion:  {frame.spherical_distortion:.4f} m\n'
        ).format(frame=self)


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
    utm = FrameConfig.T(default=FrameConfig(),
                        help='Scene Frame configuration.')
    quadtree = QuadtreeConfig.T(default=QuadtreeConfig(),
                                help='Quadtree configuration.')


class Scene(object):
    """Scene holding satellite LOS ground dispacements measurements

    :param displacement: ``NxM`` matrix of displacement in LOS
    :type displacement: :py:class:`numpy.ndrray`
    :param theta: ``NxM`` matrix of theta towards LOS.
        Theta is look vector elevation angle towards satellite from vertical
        in radians - ``pi/2: up; -pi/2: down``
    :type theta: :py:class:`numpy.ndarray`
    :param phi: ``NxM`` matrix of phi towards LOS.
        Phi look vector orientation angle towards satellite from East
        in radians.
        ``(0: east, pi/2 north)``
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
        self._log = logging.getLogger('Scene')
        self.sceneChanged = Subject()

        self._displacement = None
        self._phi = None
        self._theta = None
        self.cols = 0
        self.rows = 0
        self.los = LOSUnitVectors(scene=self)
        self.frame = Frame(scene=self, config=self.config.utm)

    @property
    def displacement(self):
        """InSAR scene displacement ``NxM`` matrix of type
        :py:class:`numpy:ndarray`
        """
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self.rows, self.cols = self._displacement.shape
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
        from kite.plot2d import ScenePlot
        return ScenePlot(self)

    def spool(self):
        """Start the spool GUI :py:class:`kite.spool.Spool` to inspect
        the scene.
        """
        if self.displacement is None:
            raise SceneError('Can not display an empty scene.')
        from kite.spool import Spool
        return Spool(scene=self)

    @classmethod
    def import_file(cls, path, **kwargs):
        """Import displacement data from foreign file format.

        :param path: Filename of resource to import
        :type path: str
        :param **kwargs: keyword arguments passed to import function
        :type **kwargs: dict
        :returns: Scene from path
        :rtype: {:py:class:`kite.Scene`}
        :raises: TypeError
        """
        import os

        if not os.path.isfile(path) or os.path.isdir(path):
            raise ImportError('File %s does not exist!' % path)

        scene = cls()
        data = None

        for mod in scene_io.__all__:
            module = eval('scene_io.%s()' % mod)
            if module.validate(path, **kwargs):
                scene._log.info('Importing %s using %s' %
                                (path, mod))
                data = module.read(path, **kwargs)
                break
        if data is None:
            raise ImportError('Could not recognize format for %s' % path)
        scene.theta = data['theta']
        scene.phi = data['phi']
        scene.displacement = data['displacement']
        scene.frame.llLat = data['llLat']
        scene.frame.llLon = data['llLon']
        scene.frame.dLat = data['dLat']
        scene.frame.dLon = data['dLon']

        scene._testImport()
        return scene

    import_file.__func__.__doc__ += \
        '\nSupported import modules: %s.\n' % (', ').join(scene_io.__all__)
    for mod in scene_io.__all__:
        import_file.__func__.__doc__ += '\n**%s**\n\n' % mod
        import_file.__func__.__doc__ += eval('scene_io.%s.__doc__' % mod)

    def _testImport(self):
        try:
            self.frame.E
            self.frame.N
            self.frame.gridE
            self.frame.gridN
            self.frame.dE
            self.frame.dN
            self.displacement
            self.theta
            self.phi
        except Exception as e:
            print e
            raise ImportError('Something went wrong during import - '
                              'see Exception!')

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
            raise ImportError('Could not load kite scene container %s'
                              % scene_name)

        scene._testImport()
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
        return num.cos(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitN(self):
        """Unit vector in North, ``NxM`` matrix like ``Scene.displacement``
        """
        return num.sin(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitU(self):
        """Unit vector Up, ``NxM`` matrix like ``Scene.displacement``
        """
        return num.cos(self._scene.theta)

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
    def createGauss(cls, nx=500, ny=500, noise=None, **kwargs):
        scene = cls()
        scene.meta.scene_title = 'Synthetic Displacement | Gaussian'
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._gaussAnomaly(scene.frame.E, scene.frame.N,
                                                 **kwargs)
        if noise is not None:
            cls._addNoise(scene, noise)
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
    def createSine(cls, nx=500, ny=500, noise=None, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Displacement | Sine'
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._sineAnomaly(scene.frame.E, scene.frame.N,
                                                **kwargs)
        if noise is not None:
            cls._addNoise(scene, noise)
        return scene

    @staticmethod
    def _prepareSceneTest(scene, nx=500, ny=500):
        scene.frame.dLat = 1e-4
        scene.frame.dLon = 1e-4
        scene.frame.E = num.arange(nx) * 50.
        scene.frame.N = num.arange(ny) * 50.
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, nx), ny).reshape((nx, ny))
        scene.phi = num.rot90(scene.theta)
        return scene

    @staticmethod
    def _addNoise(scene, noise_amplitude):
        rand = num.random.RandomState()
        noise = rand.rand(*scene.displacement.shape) *\
            num.max(scene.displacement) * noise_amplitude
        scene.displacement += noise

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
        print gauss_anomaly.shape

        return gauss_anomaly


__all__ = ['Scene', 'SceneConfig']


if __name__ == '__main__':
    testScene = SceneTest.createGauss()
