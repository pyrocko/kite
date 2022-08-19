#!/usr/bin/env python3
import copy
import hashlib
import logging
import os.path as op
import time
from datetime import datetime as dt

import numpy as num
import utm
from pyrocko.dataset.topo import srtmgl3
from pyrocko.guts import Dict, Float, Object, String, StringChoice, Timestamp, load
from pyrocko.orthodrome import latlon_to_ne, latlon_to_ne_numpy, ne_to_latlon  # noqa
from scipy import interpolate

from kite import scene_io
from kite.aps import APS, APSConfig
from kite.covariance import CovarianceConfig
from kite.deramp import Deramp, DerampConfig
from kite.gacos import GACOSConfig, GACOSCorrection
from kite.quadtree import QuadtreeConfig
from kite.scene_mask import PolygonMask, PolygonMaskConfig
from kite.util import Subject, property_cached


def read(filename):
    try:
        return Scene.load(filename)
    except (ImportError, UserIOWarning):
        pass
    try:
        scene = Scene()
        scene.import_data(filename)
        return scene
    except ImportError:
        pass
    raise ImportError("Could not read file %s" % filename)


def _setDataNumpy(obj, variable, value):
    if isinstance(value, num.ndarray):
        return obj.__setattr__(variable, value)
    else:
        raise TypeError("value must be of type numpy.ndarray")


class UserIOWarning(UserWarning):
    pass


class SceneError(Exception):
    pass


class FrameConfig(Object):
    """Config object holding :class:`kite.scene.Scene` configuration"""

    llLat = Float.T(default=0.0, help="Scene latitude of lower left corner")
    llLon = Float.T(default=0.0, help="Scene longitude of lower left corner")
    dN = Float.T(default=25.0, help="Scene pixel spacing in north, give [m] or [deg]")
    dE = Float.T(default=25.0, help="Scene pixel spacing in east, give [m] or [deg]")
    spacing = StringChoice.T(
        choices=("degree", "meter"), default="meter", help="Unit of pixel space"
    )

    def __init__(self, *args, **kwargs):
        self.old_import = False
        mapping = {"dE": "dLon", "dN": "dLat"}

        for new, old in mapping.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
                kwargs["spacing"] = "degree"
                self.old_import = True

        Object.__init__(self, *args, **kwargs)


class Frame(object):
    """Frame holding geographical references for :class:`kite.scene.Scene`

    The pixel spacing is given by ``dE`` and ``dN`` which can meters or degree.
    """

    def __init__(self, scene, config=None):
        self.evChanged = Subject()
        self._scene = scene
        self._log = scene._log.getChild("Frame")

        self.N = None
        self.E = None

        self.llEutm = None
        self.llNutm = None
        self.utm_zone = None
        self.utm_zone_letter = None
        self._meter_grid = None

        self._updateConfig(config or FrameConfig())
        self._scene.evConfigChanged.subscribe(self._updateConfig)
        self._scene.evChanged.subscribe(self.updateExtent)

    def _updateConfig(self, config=None):
        if config is not None:
            self.config = config
        elif self.config != self._scene.config.frame:
            self.config = self._scene.config.frame
        else:
            return

        if self.config.old_import:
            self._log.warning("Importing an old kite format...")
            self._log.warning("Please check your pixel spacing - dE, dN!")
        self.updateExtent()

    def updateExtent(self):
        if self._scene.cols == 0 or self._scene.rows == 0:
            return

        self.cols = self._scene.cols
        self.rows = self._scene.rows

        (
            self.llEutm,
            self.llNutm,
            self.utm_zone,
            self.utm_zone_letter,
        ) = utm.from_latlon(self.llLat, self.llLon)

        self.E = None
        self.N = None

        self.gridE = None
        self.gridN = None
        self._meter_grid = None
        self.coordinates = None

        self.config.regularize()
        self.evChanged.notify()

    @property
    def llLat(self):
        return self.config.llLat

    @llLat.setter
    def llLat(self, llLat):
        self.config.llLat = llLat
        self.updateExtent()

    @property
    def llLon(self):
        return self.config.llLon

    @llLon.setter
    def llLon(self, llLon):
        self.config.llLon = llLon
        self.updateExtent()

    @property
    def dN(self):
        return self.config.dN

    @dN.setter
    def dN(self, dN):
        self.config.dN = dN
        self.updateExtent()

    @property
    def dE(self):
        return self.config.dE

    @dE.setter
    def dE(self, dE):
        self.config.dE = dE
        self.updateExtent()

    @property
    def dEmeter(self):
        if self.isMeter():
            return self.dE

        _, dEmeter = latlon_to_ne(
            self.llLat, self.llLon, self.llLat, self.llLon + self.dE * self.cols
        )
        return dEmeter / self.cols

    @property
    def dNmeter(self):
        if self.isMeter():
            return self.dN
        dNmeter, _ = latlon_to_ne(
            self.llLat, self.llLon, self.llLat + self.dN * self.rows, self.llLon
        )
        return dNmeter / self.rows

    @property
    def dEdegree(self):
        if self.isDegree():
            return self.dE

        lat, lon = ne_to_latlon(self.llLat, self.llLon, 0.0, self.dE * self.cols)
        distLon = lon - self.llLon
        return distLon / self.cols

    @property
    def dNdegree(self):
        if self.isDegree():
            return self.dE

        lat, lon = ne_to_latlon(self.llLat, self.llLon, self.dN * self.rows, 0.0)
        distLat = lat - self.llLat
        return distLat / self.rows

    @property
    def spacing(self):
        return self.config.spacing

    @spacing.setter
    def spacing(self, unit):
        self.config.spacing = unit

    @property_cached
    def E(self):
        return num.arange(self.cols) * self.dE

    @property_cached
    def Emeter(self):
        return num.arange(self.cols) * self.dEmeter

    @property_cached
    def N(self):
        return num.arange(self.rows) * self.dN

    @property
    def lengthE(self):
        return self.cols * self.dE

    @property
    def lengthN(self):
        return self.rows * self.dN

    @property_cached
    def Nmeter(self):
        return num.arange(self.rows) * self.dNmeter

    @property_cached
    def gridE(self):
        """Grid holding local east coordinates of all pixels in ``NxM`` matrix
            of :attr:`~kite.Scene.displacement`.

        :type: :class:`numpy.ndarray`, size ``NxM``
        """
        valid_data = num.isnan(self._scene.displacement)
        gridE = num.repeat(self.E[num.newaxis, :], self.rows, axis=0)
        return num.ma.masked_array(gridE, valid_data, fill_value=num.nan)

    @property_cached
    def gridN(self):
        """Grid holding local north coordinates of all pixels in ``NxM`` matrix
            of :attr:`~kite.Scene.displacement`.

        :type: :class:`numpy.ndarray`, size ``NxM``
        """
        valid_data = num.isnan(self._scene.displacement)
        gridN = num.repeat(self.N[:, num.newaxis], self.cols, axis=1)
        return num.ma.masked_array(gridN, valid_data, fill_value=num.nan)

    def _calculateMeterGrid(self):
        if self.isMeter():
            raise ValueError(
                "Frame is defined in meter! " "Use gridE and gridN for meter grids"
            )

        if self._meter_grid is None:
            self._log.debug("Transforming latlon grid to meters...")
            gridN, gridE = latlon_to_ne_numpy(
                self.llLat,
                self.llLon,
                self.llLat + self.gridN.data.ravel(),
                self.llLon + self.gridE.data.ravel(),
            )

            valid_data = num.isnan(self._scene.displacement)
            gridE = num.ma.masked_array(
                gridE.reshape(self.gridE.shape), valid_data, fill_value=num.nan
            )
            gridN = num.ma.masked_array(
                gridN.reshape(self.gridN.shape), valid_data, fill_value=num.nan
            )
            self._meter_grid = (gridE, gridN)

        return self._meter_grid

    @property_cached
    def gridEmeter(self):
        if self.isMeter():
            return self.gridE

        return self._calculateMeterGrid()[0]

    @property_cached
    def gridNmeter(self):
        if self.isMeter():
            return self.gridN
        return self._calculateMeterGrid()[1]

    @property_cached
    def coordinates(self):
        """Local east and north coordinates of all pixels in
           ``Nx2`` matrix.

        :type: :class:`numpy.ndarray`, size ``Nx2``
        """
        coords = num.empty((self.rows * self.cols, 2))
        coords[:, 0] = num.repeat(self.E[num.newaxis, :], self.rows, axis=0).flatten()
        coords[:, 1] = num.repeat(self.N[:, num.newaxis], self.cols, axis=1).flatten()

        if self.isMeter():
            coords = ne_to_latlon(self.llLat, self.llLon, *coords.T)
            coords = num.array(coords).T

        else:
            coords[:, 0] += self.llLon
            coords[:, 1] += self.llLat

        return coords

    @property_cached
    def coordinatesMeter(self):
        """Local east and north coordinates [m] of all pixels in
           ``NxM`` matrix.

        :type: :class:`numpy.ndarray`, size ``NxM``
        """
        coords = num.empty((self.rows * self.cols, 2))
        coords[:, 0] = num.repeat(
            self.Emeter[num.newaxis, :], self.rows, axis=0
        ).flatten()
        coords[:, 1] = num.repeat(
            self.Nmeter[:, num.newaxis], self.cols, axis=1
        ).flatten()
        return coords

    def mapENMatrix(self, E, N):
        """Local map coordinates in east and north to matrix
            row and column

        :param E: Easting in local coordinates
        :type E: float
        :param N: Northing in local coordinates
        :type N: float
        :returns: Row and column
        :rtype: tuple (int), (row, column)
        """
        row = round(E / self.dE) if E > 0 else 0
        col = round(N / self.dN) if N > 0 else 0
        return int(row), int(col)

    @property
    def shape(self):
        return self._scene.shape

    def isMeter(self):
        return self.config.spacing == "meter"

    def isDegree(self):
        return self.config.spacing == "degree"

    @property
    def npixel(self):
        return self.cols * self.rows

    def __eq__(self, other):
        return (
            self.llLat == other.llLat
            and self.llLon == other.llLon
            and self.dE == other.dE
            and self.dN == other.dN
            and self.rows == other.rows
            and self.cols == other.cols
        )


class Meta(Object):
    """Meta configuration for ``Scene``."""

    scene_title = String.T(default="Unnamed Scene", help="Scene title")
    scene_id = String.T(default="None", help="Scene identification")
    satellite_name = String.T(
        default="Undefined Mission", help="Satellite mission name"
    )
    wavelength = Float.T(optional=True, help="Wavelength in [m]")
    orbital_node = StringChoice.T(
        choices=["Ascending", "Descending", "Undefined"],
        default="Undefined",
        help="Orbital direction, ascending/descending",
    )
    time_master = Timestamp.T(
        default=1481116161.930574, help="Timestamp for master acquisition"
    )
    time_slave = Timestamp.T(
        default=1482239325.482, help="Timestamp for slave acquisition"
    )
    extra = Dict.T(default={}, help="Extra header information")
    filename = String.T(optional=True)

    def __init__(self, *args, **kwargs):
        self.old_import = False

        mapping = {"orbit_direction": "orbital_node"}

        for old, new in mapping.items():
            if old in kwargs.keys():
                kwargs[new] = kwargs.pop(old, None)
                self.old_import = True

        Object.__init__(self, *args, **kwargs)

    @property
    def time_separation(self):
        """
        :getter: Absolute time difference between ``time_master``
                 and ``time_slave``
        :type: timedelta
        """
        return dt.fromtimestamp(self.time_slave) - dt.fromtimestamp(self.time_master)


class SceneConfig(Object):
    """Configuration object, gathering ``kite.Scene`` and
    sub-objects configuration.
    """

    meta = Meta.T(default=Meta.D(), help="Scene metainformation")
    frame = FrameConfig.T(default=FrameConfig.D(), help="Frame/reference configuration")
    quadtree = QuadtreeConfig.T(default=QuadtreeConfig.D(), help="Quadtree parameters")
    covariance = CovarianceConfig.T(
        default=CovarianceConfig.D(), help="Covariance parameters"
    )
    aps = APSConfig.T(default=APSConfig.D(), help="Empirical APS correction")
    gacos = GACOSConfig.T(default=GACOSConfig.D(), help="GACOS APS correction")
    polygon_mask = PolygonMaskConfig.T(
        default=PolygonMaskConfig.D(), help="Displacement mask polygon"
    )
    deramp = DerampConfig.T(default=DerampConfig.D(), help="Displacement deramp config")

    @property
    def old_import(self):
        return self.frame.old_import


def dynamicmethod(func):
    """Decorator for dynamic classmethod / instancemethod declaration"""

    def dynclassmethod(*args, **kwargs):
        if isinstance(args[0], Scene):
            return func(*args, **kwargs)
        else:
            return func(Scene(), *args, **kwargs)

    dynclassmethod.__doc__ = func.__doc__
    dynclassmethod.__name__ = func.__name__
    return dynclassmethod


class BaseScene(object):
    def __init__(self, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)

        self.evChanged = Subject()
        self.evConfigChanged = Subject()

        self._displacement = None
        self._displacement_px_var = None
        self._phi = None
        self._theta = None
        self._los_factors = None
        self.cols = 0
        self.rows = 0
        self.los = LOSUnitVectors(scene=self)

        self._elevation = {}

        frame_config = kwargs.pop("frame_config", FrameConfig())

        for fattr in ("llLat", "llLon", "dLat", "dLon"):
            coord = kwargs.pop(fattr, None)
            if coord is not None:
                frame_config.__setattr__(fattr, coord)
        self.frame = Frame(scene=self, config=frame_config)

        for attr in ("displacement", "displacement_px_var", "theta", "phi"):
            data = kwargs.pop(attr, None)
            if data is not None:
                self.__setattr__(attr, data)

    @property
    def displacement(self):
        """Surface displacement in meter on a regular grid.

        :setter: Set the unwrapped InSAR displacement.
        :getter: Return the displacement matrix.
        :type: :class:`numpy.ndarray`, ``NxM``
        """
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, "_displacement", value)
        self.rows, self.cols = self._displacement.shape
        self.evChanged.notify()

    @property
    def displacement_px_var(self):
        """Variance of the surface displacement per pixel.
        Same dimension as displacement.

        :setter: Set standard deviation of of the displacement.
        :getter: Return the standard deviation matrix.
        :type: :class:`numpy.ndarray`, ``NxM``
        """
        return self._displacement_px_var

    @displacement_px_var.setter
    def displacement_px_var(self, value):
        self._displacement_px_var = value

    @property
    def displacement_mask(self):
        """Displacement :attr:`numpy.nan` mask

        :type: :class:`numpy.ndarray`, dtype :class:`numpy.bool`
        """
        return ~num.isfinite(self.displacement)

    @property
    def shape(self):
        return self.displacement.shape

    @property
    def phi(self):
        """Horizontal angle towards satellite :abbr:`line of sight (LOS)`
        in radians counter-clockwise from East.

        .. important ::

            Kite's convention is:

            * :math:`0` is **East**
            * :math:`\\frac{\\pi}{2}` is **North**!

        :setter: Set the phi matrix for scene's displacement, can be ``int``
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
            _setDataNumpy(self, "_phi", value)
        self.phiDeg = None
        self.los_rotation_factors = None
        self.evChanged.notify()

    @property
    def theta(self):
        """Theta is the look vector elevation angle towards satellite from
        the horizon in radians. Matrix of theta towards satellite's
        :abbr:`line of sight (LOS)`.

        .. important ::

            Kite convention!

            * :math:`-\\frac{\\pi}{2}` is **Down**
            * :math:`\\frac{\\pi}{2}` is **Up**

        :setter: Set the theta matrix for scene's displacement, can be ``int``
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
            _setDataNumpy(self, "_theta", value)
        self.thetaDeg = None
        self.los_rotation_factors = None
        self.evChanged.notify()

    @property_cached
    def thetaDeg(self):
        """LOS elevation angle in degree, ``NxM`` matrix like
            :class:`kite.Scene.theta`

        :type: :class:`numpy.ndarray`
        """
        return num.rad2deg(self.theta)

    @property_cached
    def phiDeg(self):
        """LOS horizontal orientation angle in degree,
            counter-clockwise from East,``NxM`` matrix like
            :class:`kite.Scene.phi`

        :type: :class:`numpy.ndarray`
        """
        return num.rad2deg(self.phi)

    @property_cached
    def los_rotation_factors(self):
        """ Trigonometric factors to rotate displacement matrices towards LOS

        Rotation is as follows:

        ..
            displacement_los =\
                (los_rotation_factors[:, :, 0] * -down +
                 los_rotation_factors[:, :, 1] * east +
                 los_rotation_factors[:, :, 2] * north)

        :returns: Factors for rotation
        :rtype: :class:`numpy.ndarray`, ``NxMx3``
        :raises: AttributeError
        """
        if self.theta.size != self.phi.size:
            raise AttributeError(
                "LOS angles inconsistent with provided" " coordinate shape."
            )
        if self._los_factors is None:
            self._los_factors = num.empty((self.theta.shape[0], self.theta.shape[1], 3))
            self._los_factors[:, :, 0] = num.sin(self.theta)
            self._los_factors[:, :, 1] = num.cos(self.theta) * num.cos(self.phi)
            self._los_factors[:, :, 2] = num.cos(self.theta) * num.sin(self.phi)
        return self._los_factors

    def get_elevation(self, interpolation="nearest_neighbor"):
        assert interpolation in ("nearest_neighbor", "bivariate")

        if self._elevation.get(interpolation, None) is None:
            self._log.debug("Getting elevation...")
            # region = llLon, urLon, llLat, urLon
            coords = self.frame.coordinates
            lons = coords[:, 0]
            lats = coords[:, 1]

            region = (lons.min(), lons.max(), lats.min(), lats.max())
            if not srtmgl3.covers(region):
                raise AssertionError("Region is outside of SRTMGL3 topo dataset")

            tile = srtmgl3.get(region)
            if not tile:
                raise AssertionError("Cannot get SRTMGL3 topo dataset")

            if interpolation == "nearest_neighbor":
                iy = num.rint((lats - tile.ymin) / tile.dy).astype(num.intp)
                ix = num.rint((lons - tile.xmin) / tile.dx).astype(num.intp)

                elevation = tile.data[(iy, ix)]

            elif interpolation == "bivariate":
                interp = interpolate.RectBivariateSpline(tile.y(), tile.x(), tile.data)
                elevation = interp(lats, lons, grid=False)

            elevation = elevation.reshape(self.rows, self.cols)
            self._elevation[interpolation] = elevation

        return self._elevation[interpolation]

    def __neg__(self):
        ret = copy.deepcopy(self)
        ret.displacement *= -1
        return ret

    def __add__(self, other, copy_obj=True):
        if copy_obj:
            ret = copy.deepcopy(self)
        else:
            ret = self

        if not ret.frame == other.frame:
            raise ValueError("Scene frames do not align!")
        ret.displacement += other.displacement

        tmin = (
            ret.meta.time_master
            if ret.meta.time_master < other.meta.time_master
            else other.meta.time_master
        )

        tmax = (
            ret.meta.time_slave
            if ret.meta.time_slave > other.meta.time_slave
            else other.meta.time_slave
        )

        ret.meta.time_master = tmin
        ret.meta.time_slave = tmax
        return ret

    def __sub__(self, other):
        return self.__add__(-other)

    def __isub__(self, scene):
        return self.__add__(-scene, copy_obj=False)

    def __iadd__(self, scene):
        return self.__add__(scene, copy_obj=False)


class Scene(BaseScene):
    """Scene of unwrapped InSAR ground displacements measurements

    :param config: Configuration object
    :type config: :class:`~kite.scene.SceneConfig`, optional

    Optional parameters

    :param displacement: Displacement in [m]
    :type displacement: :class:`numpy.ndarray`, NxM, optional
    :param theta: Theta look angle, see :attr:`BaseScene.theta`
    :type theta: :class:`numpy.ndarray`, NxM, optional
    :param phi: Phi look angle, see :attr:`BaseScene.phi`
    :type phi: :class:`numpy.ndarray`, NxM, optional

    :param llLat: Lower left latitude in [deg]
    :type llLat: float, optional
    :param llLon: Lower left longitude in [deg]
    :type llLon: float, optional
    :param dLat: Pixel spacing in latitude [deg or m]
    :type dLat: float, optional
    :param dLon: Pixel spacing in longitude [deg or m]
    :type dLon: float, optional
    """

    def __init__(self, config=None, **kwargs):
        self.config = config or SceneConfig()
        self.meta = self.config.meta

        BaseScene.__init__(self, frame_config=self.config.frame, **kwargs)

        # wiring special methods
        self.import_data = self._import_data

        self.aps = APS(self, self.config.aps)
        self.gacos = GACOSCorrection(self, self.config.gacos)
        self.polygon_mask = PolygonMask(self, self.config.polygon_mask)
        self.deramp = Deramp(self, self.config.deramp)

        self._proc_displacement = None

        self.processing_states = {
            self.gacos: None,
            self.aps: None,
            self.polygon_mask: None,
            self.deramp: None,
        }

    @property
    def displacement(self):
        if self.has_processing_changed() or self._proc_displacement is None:

            self._proc_displacement = self._displacement.copy()
            for plugin, state in self.processing_states.items():
                self.processing_states[plugin] = plugin.get_state_hash()
                if not plugin.is_enabled():
                    continue

                t = time.time()
                plugin.apply(self._proc_displacement)
                self._log.debug(
                    "applied %s in %.4f s", plugin.__class__.__name__, time.time() - t
                )

        return self._proc_displacement

    @displacement.setter
    def displacement(self, value):
        _setDataNumpy(self, "_displacement", value)
        self.rows, self.cols = self._displacement.shape
        self.evChanged.notify()

    @property_cached
    def quadtree(self):
        """Instantiates the scene's quadtree.

        :type: :class:`kite.quadtree.Quadtree`
        """
        self._log.debug("Creating kite.Quadtree instance")
        from kite.quadtree import Quadtree

        return Quadtree(scene=self, config=self.config.quadtree)

    @property_cached
    def covariance(self):
        """Instantiates the scene's covariance attribute.

        :type: :class:`kite.covariance.Covariance`
        """
        self._log.debug("Creating kite.Covariance instance")
        from kite.covariance import Covariance

        return Covariance(scene=self, config=self.config.covariance)

    @property_cached
    def plot(self):
        """Shows a simple plot of the scene's displacement"""
        self._log.debug("Creating kite.ScenePlot instance")
        from kite.plot2d import ScenePlot

        return ScenePlot(self)

    def has_processing_changed(self):
        for plugin, state in self.processing_states.items():
            if state != plugin.get_state_hash():
                self._log.debug(
                    "processing states changed: %s", plugin.__class__.__name__
                )
                return True
        return False

    def get_plugin_state_hash(self):
        sha = hashlib.sha1()
        for plugin, state in self.processing_states.items():
            sha.update(plugin.get_state_hash().encode())

        return sha.hexdigest()

    def get_state_hash(self):
        sha = hashlib.sha1()
        for plugin, state in self.processing_states.items():
            sha.update(plugin.get_state_hash().encode())

        sha.update(self.covariance.get_state_hash().encode())
        sha.update(self.quadtree.get_state_hash().encode())
        return sha.hexdigest()

    def spool(self):
        """Start the spool user interface :class:`~kite.spool.Spool` to inspect
        the scene.
        """
        if self.displacement is None:
            raise SceneError("Can not display an empty scene.")

        from kite.spool import spool

        spool(scene=self)

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
            self._log.exception(e)
            raise ImportError("Something went wrong during import - " "see Exception!")

    def save(self, filename=None):
        """Save kite scene to kite file structure

        Saves the current scene meta information and UTM frame to a YAML
        (``.yml``) file. Numerical data (:attr:`~kite.Scene.displacement`,
        :attr:`~kite.Scene.theta` and :attr:`~kite.Scene.phi`)
        are saved as binary files from :class:`numpy.ndarray`.

        :param filename: Filenames to save scene to, defaults to
            ' :attr:`~kite.Scene.meta.scene_id` ``_``
            :attr:`~kite.Scene.meta.scene_view`
        :type filename: str, optional
        """
        filename = filename or "%s_%s" % (self.meta.scene_id, self.meta.scene_view)
        _file, ext = op.splitext(filename)
        filename = _file if ext in [".yml", ".npz"] else filename

        components = ["_displacement", "theta", "phi"]
        self._log.debug("Saving scene data to %s.npz" % filename)

        num.savez("%s.npz" % (filename), *[getattr(self, arr) for arr in components])

        self.gacos.save(op.dirname(op.abspath(filename)))

        self.saveConfig("%s.yml" % filename)

    def saveConfig(self, filename):
        _file, ext = op.splitext(filename)
        filename = filename if ext in [".yml"] else filename + ".yml"
        self._log.debug("Saving scene config to %s" % filename)
        self.config.regularize()
        self.config.dump(filename="%s" % filename, header="kite.Scene YAML Config")

    @classmethod
    def load(cls, filename):
        """Load a kite scene from file ``filename.[npz,yml]``
        structure.

        :param filename: Filenames the scene data is saved under
        :type filename: str
        :returns: Scene object from data resources
        :rtype: :class:`~kite.Scene`
        """
        filename = op.abspath(filename)
        basename = op.splitext(filename)[0]

        try:
            data = num.load("%s.npz" % basename)
            displacement = data["arr_0"]
            theta = data["arr_1"]
            phi = data["arr_2"]
        except IOError:
            raise UserIOWarning("Could not load data from %s.npz" % basename)

        try:
            config = load(filename="%s.yml" % basename)
            config.meta.filename = filename
        except IOError:
            raise UserIOWarning("Could not load %s.yml" % basename)

        scene = cls(displacement=displacement, theta=theta, phi=phi, config=config)
        scene._log.debug("Loading from %s[.npz,.yml]", basename)

        scene.meta.filename = filename

        scene._testImport()
        return scene

    def load_config(self, filename):
        self._log.debug("Loading config from %s", filename)
        self.config = load(filename=filename)
        self.meta = self.config.meta

        self.evConfigChanged.notify()

    @dynamicmethod
    def _import_data(self, path, **kwargs):
        """Import displacement data from foreign file format.

        :param path: Filename of resource to import
        :type path: str
        :param kwargs: keyword arguments passed to import function
        :type kwargs: dict
        :returns: Scene from path
        :rtype: :class:`~kite.Scene`
        :raises: TypeError
        """
        scene = self
        if not op.isfile(path) and not op.isdir(path):
            raise ImportError("File %s does not exist!" % path)

        data = None

        for mod_name in scene_io.__all__:
            cls = getattr(__import__("kite.scene_io", fromlist=mod_name), mod_name)
            module = cls()
            if module.validate(path, **kwargs):
                scene._log.debug("Importing %s using %s module" % (path, mod_name))
                data = module.read(path, **kwargs)
                break
        if data is None:
            raise ImportError("Could not recognize format for %s" % path)

        scene.meta.filename = op.abspath(path)
        return scene._import_from_dict(scene, data)

    _class_list = map("* :class:`~kite.scene_io.{}`".format, scene_io.__all__)
    _import_data.__doc__ += (
        "\nSupported import for unwrapped InSAR data are:\n\n{}\n".format(
            "\n".join(_class_list)
        )
    )
    for mod_name in scene_io.__all__:
        cls = getattr(__import__("kite.scene_io", fromlist=mod_name), mod_name)

        _import_data.__doc__ += "\n**{name}**\n\n{doc}".format(
            name=mod_name, doc=cls.__doc__
        )
    import_data = staticmethod(_import_data)

    @staticmethod
    def _import_from_dict(scene, data):
        for sk in ["theta", "phi", "displacement"]:
            setattr(scene, sk, data[sk])

        for fk, fv in data["frame"].items():
            setattr(scene.frame, fk, fv)

        for mk, mv in data["meta"].items():
            if mv is not None:
                setattr(scene.meta, mk, mv)
        scene.meta.extra.update(data["extra"])
        scene.frame.updateExtent()

        scene._testImport()
        return scene

    def __str__(self):
        return self.config.__str__()


class LOSUnitVectors(object):
    """Decompose line-of-sight (LOS) angles derived from
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
        """Unit vector east component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return self._scene.los_rotation_factors[:, :, 1]

    @property_cached
    def unitN(self):
        """Unit vector north component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return self._scene.los_rotation_factors[:, :, 2]

    @property_cached
    def unitU(self):
        """Unit vector vertical (up) component, ``NxM`` matrix like
            :attr:`~kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return self._scene.los_rotation_factors[:, :, 0]


class TestScene(Scene):
    """Test scenes for synthetic displacement"""

    @classmethod
    def createGauss(cls, nx=512, ny=512, noise=None, **kwargs):
        scene = cls()
        scene.meta.scene_title = "Synthetic Displacement | Gaussian"
        scene = cls._prepareSceneTest(scene, nx, ny)

        scene.displacement = scene._gaussAnomaly(scene.frame.E, scene.frame.N, **kwargs)
        if noise is not None:
            cls.addNoise(noise)
        return scene

    @classmethod
    def createRandom(cls, nx=512, ny=512, **kwargs):
        scene = cls()
        scene.meta.title = "Synthetic Displacement | Uniform Random"
        scene = cls._prepareSceneTest(scene, nx, ny)

        rand_state = num.random.RandomState(seed=kwargs.pop("seed", None))
        scene.displacement = (rand_state.rand(nx, ny) - 0.5) * 2

        return scene

    @classmethod
    def createSine(
        cls, nx=512, ny=512, kE=0.0041, kN=0.0061, amplitude=1.0, noise=0.5, **kwargs
    ):
        scene = cls()
        scene.meta.title = "Synthetic Displacement | Sine"
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
            scene.addNoise(noise)
        return scene

    @classmethod
    def createFractal(
        cls,
        nE=1024,
        nN=1024,
        beta=[5.0 / 3, 8.0 / 3, 2.0 / 3],
        regime=[0.15, 0.99, 1.0],
        amplitude=1.0,
    ):
        scene = cls()
        scene.meta.title = "Synthetic Displacement | Fractal Noise (Hanssen, 2001)"
        scene = cls._prepareSceneTest(scene, nE, nN)
        if (nE + nN) % 2 != 0:
            raise ArithmeticError("Dimensions of synthetic scene must " "both be even!")

        dE, dN = (scene.frame.dE, scene.frame.dN)

        rfield = num.random.rand(nE, nN)
        spec = num.fft.fft2(rfield)

        kE = num.fft.fftfreq(nE, dE)
        kN = num.fft.fftfreq(nN, dN)
        k_rad = num.sqrt(kN[:, num.newaxis] ** 2 + kE[num.newaxis, :] ** 2)

        regime = num.array(regime)
        k0 = 0.0
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
        beta += 1.0
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

        amp[k_rad == 0.0] = amp.max()

        spec *= amplitude * num.sqrt(amp)
        disp = num.abs(num.fft.ifft2(spec))
        disp -= num.mean(disp)

        scene.displacement = disp
        return scene

    def addNoise(self, noise_amplitude=1.0, seed=None):
        rand = num.random.RandomState(seed)
        noise = rand.randn(*self.displacement.shape) * noise_amplitude
        self.displacement += noise

    @staticmethod
    def _prepareSceneTest(scene, nE=512, nN=512):
        scene.frame.llLat = 0.0
        scene.frame.llLon = 0.0
        scene.frame.dLat = 5e-4
        scene.frame.dLon = 5e-4
        # scene.frame.E = num.arange(nE) * 50.
        # scene.frame.N = num.arange(nN) * 50.
        scene.theta = num.repeat(num.linspace(0.8, 0.85, nE), nN).reshape((nE, nN))
        scene.phi = num.rot90(scene.theta)
        scene.displacement = num.zeros((nE, nN))
        return scene

    @staticmethod
    def _gaussAnomaly(
        x, y, sigma_x=0.007, sigma_y=0.005, amplitude=3.0, x0=None, y0=None
    ):
        if x0 is None:
            x0 = x.min() + abs(x.max() - x.min()) / 2
        if y0 is None:
            y0 = y.min() + abs(y.max() - y.min()) / 2
        X, Y = num.meshgrid(x, y)

        gauss_anomaly = amplitude * num.exp(
            -(((X - x0) ** 2 / 2 * sigma_x**2) + (Y - y0) ** 2 / 2 * sigma_y**2)
        )

        return gauss_anomaly


__all__ = ["Scene", "SceneConfig"]


if __name__ == "__main__":
    testScene = TestScene.createGauss()
