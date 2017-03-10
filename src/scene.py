#!/bin/python
import logging
import numpy as num
import utm
import os.path as op

from pyrocko import guts
from pyrocko.orthodrome import latlon_to_ne  # noqa
from datetime import datetime as dt
from .quadtree import QuadtreeConfig
from .covariance import CovarianceConfig
from .meta import Subject, property_cached, greatCircleDistance
from . import scene_io

logging.basicConfig(level=20)


def read(filename):
    scene = Scene()
    try:
        scene.load(filename)
        return scene
    except (ImportError, UserIOWarning):
        pass
    try:
        scene.import_data(filename)
        return scene
    except ImportError:
        pass
    raise ImportError('Could not read file %s' % filename)


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
                        help='Scene pixel spacing in x direction [deg]')
    dLon = guts.Float.T(default=1.e-3,
                        help='Scene pixel spacing in y direction [deg]')


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

        self.offsetE = 0.
        self.offsetN = 0.

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
        self.E = num.arange(self.cols) * self.dE + self.offsetE
        self.N = num.arange(self.rows) * self.dN + self.offsetN

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
        """ UTM grid holding eastings of all pixels in ``NxM`` matrix
            of :attr:`~kite.Scene.displacement`.

        :type: :class:`numpy.ndarray`, size ``NxM``
        """
        valid_data = num.isnan(self._scene.displacement)
        gridE = num.repeat(self.E[num.newaxis, :],
                           self.rows, axis=0)
        return num.ma.masked_array(gridE, valid_data, fill_value=num.nan)

    @property_cached
    def gridN(self):
        """ UTM grid holding northings of all pixels in ``NxM`` matrix
            of :attr:`~kite.Scene.displacement`.

        :type: :class:`numpy.ndarray`, size ``NxM``
        """
        valid_data = num.isnan(self._scene.displacement)
        gridN = num.repeat(self.N[:, num.newaxis],
                           self.cols, axis=1)
        return num.ma.masked_array(gridN, valid_data, fill_value=num.nan)

    def setENOffset(self, east, north):
        """Set scene offsets in local cartesian coordinates.

        :param east: East offset in [m]
        :type east: float, :class:`numpy.ndarray` or None
        :param north: North offset in [m]
        :type north: float, :class:`numpy.ndarray` or None
        """
        self.offsetE = east
        self.offsetN = north
        self._updateExtent()

    def setLatLonReference(self, lat, lon):
        pass

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
        row = int(E/self.dE) if E > 0 else 0
        col = int(N/self.dN) if N > 0 else 0
        return row, col

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
    scene_title = guts.String.T(
        default='Undefined',
        help='Scene title')
    scene_id = guts.String.T(
        default='None',
        help='Scene identification')
    satellite_name = guts.String.T(
        default='Undefined',
        help='Satellite mission name')
    wavelength = guts.Float.T(
        optional=True,
        help='Wavelength in [m]')
    orbit_direction = guts.StringChoice.T(
        choices=['Ascending', 'Descending', 'Undefined'],
        default='Undefined',
        help='Orbital direction, ascending/descending')
    time_master = guts.Timestamp.T(
        default=1481116161.930574,
        help='Timestamp for master acquisition')
    time_slave = guts.Timestamp.T(
        default=1482239325.482,
        help='Timestamp for slave acquisition')
    extra = guts.Dict.T(
        default={},
        help='Extra header information')

    @property
    def time_separation(self):
        '''
        :getter: Absolute time difference between ``time_master``
                 and ``time_slave``
        :type: timedelta
        '''
        return dt.fromtimestamp(self.time_slave) -\
            dt.fromtimestamp(self.time_master)


class SceneConfig(guts.Object):
    """ Configuration object, gathering ``kite.Scene`` and
        sub-objects configuration.
    """
    meta = Meta.T(
        default=Meta(),
        help='Scene metainformation')
    frame = FrameConfig.T(
        default=FrameConfig(),
        help='Frame/reference configuration')
    quadtree = QuadtreeConfig.T(
        default=QuadtreeConfig(),
        help='Quadtree parameters')
    covariance = CovarianceConfig.T(
        default=CovarianceConfig(),
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
    """
    evChanged = Subject()
    evConfigChanged = Subject()

    def __init__(self, config=SceneConfig()):
        self._setupLogging()
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

    def _setupLogging(self):
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
        :type: :class:`numpy.ndarray`, ``NxM``
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
        """ Displacement :attr:`numpy.nan` mask

        :type: :class:`numpy.ndarray`, dtype :class:`numpy.bool`
        """
        return num.isnan(self.displacement)

    @property
    def phi(self):
        """ Horizontal angle towards satellite' :abbr:`line of sight (LOS)`
            in radians

        .. important ::

            Kite convention is:

            * :math:`0` is **East**
            * :math:`\\frac{\\pi}{2}` is **North**!

        :setter: Set the phi matrix for scene's displacement, can be int
                 for static look vector.
        :type: :class:`numpy.ndarray`, size same as
               :attr:`~kite.Scene.displacement` or int
        """
        return self._phi

    @phi.setter
    def phi(self, value):
        if isinstance(value, float):
            self._phi = value
        else:
            _setDataNumpy(self, '_phi', value)
        self.evChanged.notify()

    @property
    def theta(self):
        """ Theta is look vector elevation angle towards satellite from horizon
            in radians. Matrix of theta towards satellite's
            :abbr:`line of sight (LOS)`.

        .. important ::

            Kite convention!

            * :math:`-\\frac{\\pi}{2}` is **Down**
            * :math:`\\frac{\\pi}{2}` is **Up**

        :setter: Set the theta matrix for scene's displacement, can be int
                 for static look vector.
        :type: :class:`numpy.ndarray`, size same as
               :attr:`~kite.Scene.displacement` or int
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        if isinstance(value, float):
            self._theta = value
        else:
            _setDataNumpy(self, '_theta', value)
        self.evChanged.notify()

    @property_cached
    def thetaDeg(self):
        """ LOS elevation angle in degree, ``NxM`` matrix like
            :class:`kite.Scene.theta`

        :type: :class:`numpy.ndarray`
        """
        return num.rad2deg(self.theta)

    @property_cached
    def phiDeg(self):
        """ LOS horizontal orientation angle in degree, ``NxM`` matrix like
            :class:`kite.Scene.theta`

        :type: :class:`numpy.ndarray`
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
        """ Start the spool user interface :class:`~kite.spool.Spool` to inspect
        the scene.
        """
        if self.displacement is None:
            raise SceneError('Can not display an empty scene.')
        from kite.spool import Spool
        spool = Spool(scene=self)
        return spool.spool_win.buildViews()

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
        (``.yml``) file. Numerical data (:attr:`~kite.Scene.displacement`,
        :attr:`~kite.Scene.theta` and :attr:`~kite.Scene.phi`)
        are saved as binary files from :class:`numpy.ndarray`.

        :param filename: Filenames to save scene to, defaults to
            ' :attr:`~kite.Scene.meta.scene_id` ``_``
            :attr:`~kite.Scene.meta.scene_view`
        :type filename: str, optional
        """
        filename = filename or '%s_%s' % (self.meta.scene_id,
                                          self.meta.scene_view)
        _file, ext = op.splitext(filename)
        filename = _file if ext in ['yml', 'npz'] else filename

        components = ['displacement', 'theta', 'phi']
        self._log.info('Saving scene data to %s.npz' % filename)

        num.savez('%s.npz' % (filename),
                  *[getattr(self, arr) for arr in components])
        self.save_config('%s.yml' % filename)

    def save_config(self, filename):
        _file, ext = op.splitext(filename)
        filename = _file if ext in ['yml'] else filename
        self._log.info('Saving scene config to %s' % filename)
        self.config.dump(filename='%s' % filename,
                         header='kite.Scene YAML Config')

    @dynamicmethod
    def _load(self, filename):
        """ Load a kite scene from file ``filename.[npz,yml]``
        structure.

        :param filename: Filenames the scene data is saved under
        :type filename: str
        :returns: Scene object from data resources
        :rtype: :class:`~kite.Scene`
        """
        scene = self
        components = ['displacement', 'theta', 'phi']

        basename = op.splitext(filename)[0]
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

        scene.meta.filename = op.basename(filename)
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
        :param kwargs: keyword arguments passed to import function
        :type kwargs: dict
        :returns: Scene from path
        :rtype: :class:`~kite.Scene`
        :raises: TypeError
        """
        scene = self
        if not op.isfile(path) or op.isdir(path):
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

        for sk in ['theta', 'phi', 'displacement']:
            setattr(scene, sk, data[sk])

        for fk, fv in data['frame'].iteritems():
            setattr(scene.frame, fk, fv)

        for mk, mv in data['meta'].iteritems():
            if mv is not None:
                setattr(scene.meta, mk, mv)
        scene.meta.extra.update(data['extra'])

        scene.meta.filename = op.basename(path)
        scene._testImport()
        return scene

    _import_data.__doc__ += \
        '\nSupported import modules are **%s**.\n'\
        % (', ').join(scene_io.__all__)
    for mod in scene_io.__all__:
        _import_data.__doc__ += '\n**%s**\n\n' % mod
        _import_data.__doc__ += eval('scene_io.%s.__doc__' % mod)
    import_data = staticmethod(_import_data)

    def __str__(self):
        return self.config.__str__()


class LOSUnitVectors(object):
    """ Decompose line-of-sight (LOS) angles derived from
    :attr:`~kite.Scene.displacement` to unit vector.
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
        """ Unit vector east component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return num.cos(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitN(self):
        """ Unit vector north component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return num.sin(self._scene.phi) * num.sin(self._scene.theta)

    @property_cached
    def unitU(self):
        """ Unit vector vertical (up) component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return num.cos(self._scene.theta)


class SceneTest(Scene):
    """Test scenes for synthetic displacements """

    @classmethod
    def createGauss(cls, nx=512, ny=512, noise=None, **kwargs):
        scene = cls()
        scene.meta.scene_title = 'Synthetic Displacement | Gaussian'
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._gaussAnomaly(scene.frame.E, scene.frame.N,
                                                 **kwargs)
        if noise is not None:
            cls.addNoise(noise)
        return scene

    @classmethod
    def createRandom(cls, nx=512, ny=512, **kwargs):
        scene = cls()
        scene.meta.title = 'Synthetic Displacement | Uniform Random'
        scene = cls._prepareSceneTest(scene, nx, ny)

        rand_state = num.random.RandomState(seed=1010)
        scene.displacement = (rand_state.rand(nx, ny)-.5)*2

        return scene

    @classmethod
    def createSine(cls, nx=512, ny=512, kE=.0041, kN=.0061, amplitude=1.,
                   noise=.5, **kwargs):
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
            cls.addNoise(noise)
        return scene

    @classmethod
    def createFractal(cls, nE=1024, nN=1024,
                      beta=[5./3, 8./3, 2./3], regime=[.15, .99, 1.],
                      amplitude=1.):
        scene = cls()
        scene.meta.title =\
            'Synthetic Displacement | Fractal Noise (Hanssen, 2001)'
        scene = cls._prepareSceneTest(scene, nE, nN)
        if (nE+nN) % 2 != 0:
            raise ArithmeticError('Dimensions of synthetic scene must '
                                  'both be even!')

        dE, dN = (scene.frame.dE, scene.frame.dN)

        rfield = num.random.rand(nE, nN)
        spec = num.fft.fft2(rfield)

        kE = num.fft.fftfreq(nE, dE)
        kN = num.fft.fftfreq(nN, dN)
        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

        regime = num.array(regime)
        k0 = 0.
        k1 = regime[0] * k_rad.max()
        k2 = regime[1] * k_rad.max()

        r0 = num.logical_and(k_rad > k0, k_rad < k1)
        r1 = num.logical_and(k_rad >= k1, k_rad < k2)
        r2 = k_rad >= k2

        beta = num.array(beta)
        # From Hanssen (2001)
        #   beta+1 is used as beta, since, the power exponent
        #   is defined for a 1D slice of the 2D spectrum:
        #   austin94: "Adler, 1981, shows that the surface profile
        #   created by the intersection of a plane and a
        #   2-D fractal surface is itself fractal with
        #   a fractal dimension  equal to that of the 2D
        #   surface decreased by one."
        beta += 1.
        # From Hanssen (2001)
        #   The power beta/2 is used because the power spectral
        #   density is proportional to the amplitude squared
        #   Here we work with the amplitude, instead of the power
        #   so we should take sqrt( k.^beta) = k.^(beta/2)  RH
        # beta /= 2.

        amp = num.zeros_like(k_rad)
        amp[r0] = k_rad[r0] ** -beta[0]
        amp[r0] /= amp[r0].max()

        amp[r1] = k_rad[r1] ** -beta[1]
        amp[r1] /= amp[r1].max() / amp[r0].min()

        amp[r2] = k_rad[r2] ** -beta[2]
        amp[r2] /= amp[r2].max() / amp[r1].min()

        amp[k_rad == 0.] = amp.max()

        spec *= amplitude * num.sqrt(amp)
        disp = num.abs(num.fft.ifft2(spec))
        disp -= num.mean(disp)

        scene.displacement = disp
        return scene

    def addNoise(self, noise_amplitude):
        rand = num.random.RandomState()
        noise = rand.randn(*self.displacement.shape) * noise_amplitude
        self.displacement += noise

    @staticmethod
    def _prepareSceneTest(scene, nE=512, nN=512):
        scene.frame.llLat = 0.
        scene.frame.llLon = 0.
        scene.frame.dLat = 5e-4
        scene.frame.dLon = 5e-4
        # scene.frame.E = num.arange(nE) * 50.
        # scene.frame.N = num.arange(nN) * 50.
        scene.theta = num.repeat(
            num.linspace(0.8, 0.85, nE), nN).reshape((nE, nN))
        scene.phi = num.rot90(scene.theta)
        scene.displacement = num.zeros((nE, nN))
        return scene

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
