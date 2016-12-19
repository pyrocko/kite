#!/bin/python
import logging
import numpy as num
import utm

from pyrocko import guts
from .quadtree import QuadtreeConfig
from .covariance import CovarianceConfig
from .meta import Subject, property_cached, greatCircleDistance
from . import scene_io
from os import path
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


class FrameConfig(guts.Object):
    """
    Config object holding :class:`kite.scene.Scene` cobfiguration
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
    """ UTM frame holding geographical references for :class:`kite.scene.Scene`
    """
    evChanged = Subject()

    def __init__(self, scene, config=FrameConfig()):
        self._scene = scene
        self._log = scene._log.getChild('Frame')

        self.extentE = 0.
        self.extentN = 0.
        self.spherical_distortion = 0.
        self.urE = 0.
        self.urN = 0.
        self.dN = 0.
        self.dE = 0.
        self.llEutm = None
        self.llNutm = None
        self.utm_zone = None
        self.llN = None
        self.llE = None
        self.N = None
        self.E = None

        self._updateConfig(config)
        self._scene.evConfigChanged.subscribe(self._updateConfig)
        self._scene.evChanged.subscribe(self._updateExtent)

    def _updateConfig(self, config=None):
        if config is not None:
            self.config = config
        elif self.config != self._scene.config.frame:
            self.config = self._scene.config.frame
        else:
            return

        self._updateExtent()
        self.evChanged.notify()

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

    def mapMatrixEN(self, row, col):
        """ Maps matrix row, column to local easting and northing.

        :param row: Matrix row number
        :type row: int
        :param col: Matrix column number
        :type col: int
        :returns: Easting and northing in local coordinates
        :rtype: tuple (float), (easting, northing)
        """
        return row * self.dE, col * self.dN

    def mapENMatrix(self, E, N):
        """ Maps local coordinates (easting and northing) to matrix
            row and column

        :param E: Easting in local coordinates
        :type E: float
        :param N: Northing in local coordinates
        :type N: float
        :returns: Row and column
        :rtype: tuple (int), (row, column)
        """
        return int(E/self.dE), int(N/self.dN)

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
    """ Meta configuration for ``Scene``.
    """
    scene_title = guts.String.T(default='Unnamed Scene')
    scene_id = guts.String.T(default='INSAR')
    orbit_direction = guts.String.T(default='ASCENDING')
    master_time = guts.Timestamp.T(default=0.0)
    slave_time = guts.Timestamp.T(default=86400.0)
    satellite_name = guts.String.T(default='Unnamed Satellite')

    @property
    def time_separation(self):
        return slave_time - master_time


class SceneConfig(guts.Object):
    """ Configuration object, gathering ``kite.Scene`` and
    sub-objects configuration.
    """
    meta = Meta.T(default=Meta(),
                  help='Scene metainformation')
    frame = FrameConfig.T(default=FrameConfig(),
                          help='Frame/reference configuration')
    quadtree = QuadtreeConfig.T(default=QuadtreeConfig(),
                                help='Quadtree parameters')
    covariance = CovarianceConfig.T(default=CovarianceConfig(),
                                    help='Covariance parameters')


def dynamicmethod(func):
    ''' Decorator for dynamic classmethod / instancemethod declaration '''
    def dynclassmethod(*args, **kwargs):
        if isinstance(args[0], Scene):
            return func(*args, **kwargs)
        else:
            return func(Scene(), *args, **kwargs)

    dynclassmethod.__doc__ = func.__doc__
    dynclassmethod.__name__ = func.__name__
    return dynclassmethod


class Scene(object):
    """Scene holding satellite LOS ground dispacements measurements

    :param displacement: ``NxM`` matrix of displacement in LOS
    :type displacement: :class:`numpy.array`
    :param theta: ``NxM`` matrix of theta towards LOS.
        Theta is look vector elevation angle towards satellite from vertical
        in radians - ``pi/2: up; -pi/2: down``
    :type theta: :class:`numpy.array`
    :param phi: ``NxM`` matrix of phi towards LOS.
        Phi look vector orientation angle towards satellite from East
        in radians.
        ``(0: east, pi/2 north)``
    :type phi: :class:`numpy.array`
    :param meta: Meta information for the scene
    :type meta: :class:`kite.scene.Meta`

    :param los: Displacement measurements (displacement, theta, phi) from
        satellite measurements
    :type los: :class:`kite.scene.DisplacementLOS`
    """
    evChanged = Subject()
    evConfigChanged = Subject()

    def __init__(self, config=SceneConfig()):
        self._setLoggingUp()
        self.config = config
        self.meta = self.config.meta

        self._displacement = None
        self._phi = None
        self._theta = None
        self.cols = 0
        self.rows = 0
        self.los = LOSUnitVectors(scene=self)
        self.frame = Frame(scene=self, config=self.config.frame)

        self.import_data = self._import_data
        self.load = self._load

    def _setLoggingUp(self):
        logging.basicConfig(level=logging.DEBUG)

        self._log = logging.getLogger('Scene')
        self._log.setLevel(logging.DEBUG)

        self._log_stream = None
        for l in self._log.parent.handlers:
            if isinstance(l, logging.StreamHandler):
                self._log_stream = l
        if self._log_stream is None:
            self._log_stream = logging.StreamHandler()
            self._log.addHandler(self._log_stream)
        self._log_stream.setLevel(logging.INFO)

        self.setLogLevel = self._log_stream.setLevel

    @property
    def displacement(self):
        """ Geodetical displacement in *meter*.

        :setter: Set the unwrapped InSAR displacement.
        :getter: Return the displacement matrix.
        :type: :class:`numpy.array`, ``NxM``
        """
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, '_displacement', value)
        self.rows, self.cols = self._displacement.shape
        self.displacement_mask = None
        self.evChanged.notify()

    @property_cached
    def displacement_mask(self):
        """
        :getter: Displacement :attr:`numpy.nan` mask
        :type: :class:`numpy.array`, dtype :class:`numpy.bool`
        """
        return num.isnan(self.displacement)

    @property
    def phi(self):
        """ Horizontal angle towards satellite' point of view in radians.
        (``0: east, pi/2: north``)

        :getter: Returns the phi angles
        :setter: Set the phi matrix for scene's displacement, can be ``int``
            for static look vector.
        :type: :class:`numpy.array` or ``int``
        """
        return self._phi

    @phi.setter
    def phi(self, value):
        if isinstance(value, float):
            self._phi = value
        else:
            _setDataNumpy(self, '_phi', value)
        self.evChanged.notify()

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
        """ ``NxM`` matrix of theta towards satellite' line of sight.
        Theta is look vector elevation angle towards satellite from horizon
        in radians - ``pi/2: up; -pi/2: down``

        :getter: Returns the theta angles
        :setter: Set the theta matrix for scene's displacement, can be ``int``
            for static look vector.
        :type: :class:`numpy.array` or `int`
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        if isinstance(value, float):
            self._theta = value
        else:
            _setDataNumpy(self, '_theta', value)
        self.evChanged.notify()

    @theta.getter
    def theta(self):
        if isinstance(self._theta, float):
            _a = num.empty_like(self.displacement)
            _a.fill(self._theta)
            return _a
        else:
            return self._theta

    @property_cached
    def thetaDeg(self):
        """ LOS incident angle in degree, ``NxM`` matrix like
            :class:`kite.Scene.theta`
        :type: :class:`numpy.array`
        """
        return num.rad2deg(self.theta)

    @property_cached
    def phiDeg(self):
        """ LOS incident angle in degree, ``NxM`` matrix like
            :class:`kite.Scene.theta`
        :type: :class:`numpy.array`
        """
        return num.rad2deg(self.phi)

    @property_cached
    def quadtree(self):
        """ Instanciates the scene's quadtree.
        :type: :class:`kite.quadtree.Quadtree`
        """
        self._log.debug('Creating kite.Quadtree instance')
        from kite.quadtree import Quadtree
        return Quadtree(scene=self, config=self.config.quadtree)

    @property_cached
    def covariance(self):
        """ Instanciates the scene's covariance attribute.
        :type: :class:`kite.covariance.Covariance`
        """
        self._log.debug('Creating kite.Covariance instance')
        from kite.covariance import Covariance
        return Covariance(scene=self, config=self.config.covariance)

    @property_cached
    def plot(self):
        ''' Shows a simple plot of the scene's displacement
        '''
        self._log.debug('Creating kite.ScenePlot instance')
        from kite.plot2d import ScenePlot
        return ScenePlot(self)

    def spool(self):
        """ Start the spool user interface :class:`kite.spool.Spool` to inspect
        the scene.
        """
        if self.displacement is None:
            raise SceneError('Can not display an empty scene.')
        from kite.spool import Spool
        return Spool(scene=self)

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

    def save(self, filename=None):
        """ Save kite scene to kite file structure

        Saves the current scene meta information and UTM frame to a YAML
        (``.yml``) file. Numerical data (:attr:`kite.Scene.displacement`,
        :attr:`kite.Scene.theta` and :attr:`kite.Scene.phi`)
        are saved as binary files from :class:`numpy.array`.

        :param filename: Filenames to save scene to, defaults to
            ' :attr:`kite.Scene.meta.scene_id` ``_``
            :attr:`kite.Scene.meta.scene_view`
        :type filename: str, optional
        """
        filename = filename or '%s_%s' % (self.meta.scene_id,
                                          self.meta.scene_view)
        _file, ext = path.splitext(filename)
        filename = _file if ext in ['yml', 'npz'] else filename

        components = ['displacement', 'theta', 'phi']
        self._log.info('Saving scene data to %s.npz' % filename)

        num.savez('%s.npz' % (filename),
                  *[getattr(self, arr) for arr in components])
        self.save_config('%s.yml' % filename)

    def save_config(self, filename):
        _file, ext = path.splitext(filename)
        filename = _file if ext in ['yml'] else filename
        self._log.info('Saving scene config to %s' % filename)
        self.config.dump(filename='%s' % filename,
                         header='kite.Scene YAML Config')

    @dynamicmethod
    def _load(self, filename):
        """ Load a kite scene from file ``filename.[npz,yml]``
        structure

        :param filename: Filenames the scene data is saved under
        :type filename: str
        :returns: Scene object from data resources
        :rtype: :class:`kite.Scene`
        """
        scene = self
        components = ['displacement', 'theta', 'phi']

        basename = path.splitext(filename)[0]
        scene._log.info('Loading from %s[.npz,.yml]' % basename)
        try:
            data = num.load('%s.npz' % basename)
            for i, comp in enumerate(components):
                scene.__setattr__(comp, data['arr_%d' % i])
        except IOError:
            raise UserIOWarning('Could not load data from %s.npz' % basename)

        try:
            scene.load_config('%s.yml' % basename)
        except IOError:
            raise UserIOWarning('Could not load %s.yml' % basename)

        scene._testImport()
        return scene

    load = staticmethod(_load)

    def load_config(self, filename):
        self._log.info('Loading config from %s' % filename)
        self.config = guts.load(filename=filename)
        self.meta = self.config.meta

        self.evConfigChanged.notify()

    @dynamicmethod
    def _import_data(self, path, **kwargs):
        """ Import displacement data from foreign file format.

        :param path: Filename of resource to import
        :type path: str
        :param **kwargs: keyword arguments passed to import function
        :type **kwargs: dict
        :returns: Scene from path
        :rtype: :class:`kite.Scene`
        :raises: TypeError
        """
        scene = self
        import os
        if not os.path.isfile(path) or os.path.isdir(path):
            raise ImportError('File %s does not exist!' % path)
        data = None

        for mod in scene_io.__all__:
            module = eval('scene_io.%s(scene)' % mod)
            if module.validate(path, **kwargs):
                scene._log.info('Importing %s using %s module' %
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

    _import_data.__doc__ += \
        '\nSupported import modules: %s.\n' % (', ').join(scene_io.__all__)
    for mod in scene_io.__all__:
        _import_data.__doc__ += '\n**%s**\n\n' % mod
        _import_data.__doc__ += eval('scene_io.%s.__doc__' % mod)
    import_data = staticmethod(_import_data)

    def __str__(self):
        return self.config.__str__()


class LOSUnitVectors(object):
    """ Decomposed Line Of Sight vectors (LOS) derived from
    ``Scene.displacement``.
    """
    def __init__(self, scene):
        self._scene = scene
        self._scene.evChanged.subscribe(self._flush_vectors)

    def _flush_vectors(self):
        self.unitE = None
        self.unitN = None
        self.unitU = None

    @property_cached
    def unitE(self):
        """ Unit vector in East, ``NxM`` matrix like
            :attr:`kite.Scene.displacement`
        :type: :class:`numpy.array`
        """
        return num.cos(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitN(self):
        """ Unit vector in North, ``NxM`` matrix like
            :attr:`kite.Scene.displacement`
        :type: :class:`numpy.array`
        """
        return num.sin(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitU(self):
        """ Unit vector Up, ``NxM`` matrix like
            :attr:`kite.Scene.displacement`
        :type: :class:`numpy.array`
        """
        return num.cos(self._scene.theta)


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
            cls.addNoise(scene, noise)
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
    def createSine(cls, nx=500, ny=500, kE=.0041, kN=.0061, amplitude=3.,
                   noise=1., **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Displacement | Sine'
        scene = cls._prepareSceneTest(scene, nx, ny)

        E, N = num.meshgrid(scene.frame.E, scene.frame.N)
        displ = num.zeros_like(E)

        kE = num.random.rand(3) * kE
        kN = num.random.rand(3) * kN

        for ke in kE:
            phase = num.random.randn(1)[0]
            displ += num.sin(ke * E + phase)
        for kn in kN:
            phase = num.random.randn(1)[0]
            displ += num.sin(kn * N + phase)
        displ -= num.mean(displ)

        scene.displacement = displ * amplitude
        if noise is not None:
            cls.addNoise(scene, noise)
        return scene

    @staticmethod
    def _prepareSceneTest(scene, nx=500, ny=500):
        scene.frame.llLat = 32.1
        scene.frame.llLon = 54.14
        scene.frame.dLat = 1e-4
        scene.frame.dLon = 1e-4
        scene.frame.E = num.arange(nx) * 50.
        scene.frame.N = num.arange(ny) * 50.
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, nx), ny).reshape((nx, ny))
        scene.phi = num.rot90(scene.theta)
        return scene

    @staticmethod
    def addNoise(scene, noise_amplitude):
        rand = num.random.RandomState()
        noise = rand.randn(*scene.displacement.shape) * noise_amplitude
        scene.displacement += noise

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
