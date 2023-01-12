import time
from hashlib import sha1

import numpy as np
from pyrocko import guts
from pyrocko import orthodrome as od

from .util import Subject, derampMatrix, property_cached


class QuadNode(object):
    """A node (or *tile*) in held by :class:`~kite.Quadtree`. Each node in the
    tree hold a back reference to the quadtree and scene to access

    :param llr: Lower left corner row in :attr:`kite.Scene.displacement`
        matrix.
    :type llr: int
    :param llc: Lower left corner column in :attr:`kite.Scene.displacement`
        matrix.
    :type llc: int
    :param length: Length of node in from ``llr, llc`` in both dimensions
    :type length: int
    :param id: Unique id of node
    :type id: str
    :param children: Node's children
    :type children: List of :class:`~kite.quadtree.QuadNode`
    """

    CORNERS = (0, 0), (0, 1), (1, 0), (1, 1)
    MIN_PIXEL_LENGTH_NODE = None

    def __init__(self, quadtree, displacement, llr, llc, length):
        self.children = []
        self.llr = int(llr)
        self.llc = int(llc)
        self.length = int(length)
        self._slice_rows = slice(self.llr, self.llr + self.length)
        self._slice_cols = slice(self.llc, self.llc + self.length)
        self.id = "node_%d-%d_%d" % (self.llr, self.llc, self.length)

        self.quadtree = quadtree
        self._displacement = displacement
        self.scene = quadtree.scene
        self.frame = quadtree.frame

    @property_cached
    def nan_fraction(self):
        """Fraction of NaN values within the tile
        :type: float
        """
        return float(np.sum(self.displacement_mask)) / self.displacement.size

    @property_cached
    def npixel(self):
        return self.displacement.size

    @property_cached
    def mean(self):
        """Mean displacement
        :type: float
        """
        return float(np.nanmean(self.displacement))

    @property_cached
    def median(self):
        """Median displacement
        :type: float
        """
        return float(np.nanmedian(self.displacement))

    @property_cached
    def std(self):
        """Standard deviation of displacement
        :type: float
        """
        return float(np.nanstd(self.displacement))

    @property_cached
    def var(self):
        """Variance of displacement
        :type: float
        """
        return float(np.nanvar(self.displacement))

    @property_cached
    def mean_px_var(self):
        """Variance of displacement
        :type: float
        """
        if self.displacement_px_var is not None:
            return float(np.nanmean(self.displacement_px_var))
        return None

    @property_cached
    def corr_median(self):
        """Standard deviation of node's displacement corrected for median
        :type: float
        """
        return float(np.nanstd(self.displacement - self.median))

    @property_cached
    def corr_mean(self):
        """Standard deviation of node's displacement corrected for mean
        :type: float
        """
        return float(np.nanstd(self.displacement - self.mean))

    @property_cached
    def corr_bilinear(self):
        """Standard deviation of node's displacement corrected for bilinear
            trend (2D)
        :type: float
        """
        return float(np.nanstd(derampMatrix(self.displacement)))

    @property
    def weight(self):
        """
        :getter: Absolute weight derived from :class:`kite.Covariance`
         - works on tree leaves only.
        :type: float
        """
        return float(self.scene.covariance.getLeafWeight(self))

    @property_cached
    def focal_point(self):
        """Node focal point in local coordinates respecting NaN values
        :type: tuple, float - (easting, northing)
        """
        E = float(np.mean(self.gridE.compressed()) + self.frame.dE / 2)
        N = float(np.mean(self.gridN.compressed()) + self.frame.dN / 2)
        return E, N

    @property_cached
    def focal_point_meter(self):
        """Node focal point in local coordinates respecting NaN values
        :type: tuple, float - (easting, northing)
        """
        E = float(np.mean(self.gridEmeter.compressed() + self.frame.dEmeter / 2))
        N = float(np.mean(self.gridNmeter.compressed() + self.frame.dNmeter / 2))
        return E, N

    @property_cached
    def displacement(self):
        """Displacement array, slice from :attr:`kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        return self._displacement[self._slice_rows, self._slice_cols]

    @property_cached
    def displacement_masked(self):
        """Masked displacement,
            see :attr:`~kite.quadtree.QuadNode.displacement`
        :type: :class:`numpy.ndarray`
        """
        return np.ma.masked_array(
            self.displacement, self.displacement_mask, fill_value=np.nan
        )

    @property_cached
    def displacement_mask(self):
        """Displacement nan mask of
            :attr:`~kite.quadtree.QuadNode.displacement`
        :type: :class:`numpy.ndarray`, dtype :class:`numpy.bool`

        .. todo ::

            Faster to slice Scene.displacement_mask?
        """
        return np.isnan(self.displacement)

    @property_cached
    def displacement_px_var(self):
        """Displacement array, slice from :attr:`kite.Scene.displacement`
        :type: :class:`numpy.ndarray`
        """
        if self.scene.displacement_px_var is not None:
            return self.scene.displacement_px_var[self._slice_rows, self._slice_cols]
        return None

    @property_cached
    def phi(self):
        """Median Phi angle, see :class:`~kite.Scene`.
        :type: float
        """
        phi = self.scene.phi[self._slice_rows, self._slice_cols]
        return np.nanmedian(phi[~self.displacement_mask])

    @property_cached
    def theta(self):
        """Median Theta angle, see :class:`~kite.Scene`.
        :type: float
        """
        theta = self.scene.theta[self._slice_rows, self._slice_cols]
        return np.nanmedian(theta[~self.displacement_mask])

    @property
    def unitE(self):
        unitE = self.scene.los_rotation_factors[self._slice_rows, self._slice_cols, 1]
        return np.nanmedian(unitE[~self.displacement_mask])

    @property
    def unitN(self):
        unitN = self.scene.los_rotation_factors[self._slice_rows, self._slice_cols, 2]
        return np.nanmedian(unitN[~self.displacement_mask])

    @property
    def unitU(self):
        unitU = self.scene.los_rotation_factors[self._slice_rows, self._slice_cols, 0]
        return np.nanmedian(unitU[~self.displacement_mask])

    @property_cached
    def gridE(self):
        """Grid holding local east coordinates,
            see :attr:`kite.scene.Frame.gridE`.
        :type: :class:`numpy.ndarray`
        """
        return self.scene.frame.gridE[self._slice_rows, self._slice_cols]

    @property_cached
    def gridEmeter(self):
        """Grid holding local east coordinates,
            see :attr:`kite.scene.Frame.gridEmeter`.
        :type: :class:`numpy.ndarray`
        """
        return self.scene.frame.gridEmeter[self._slice_rows, self._slice_cols]

    @property_cached
    def gridN(self):
        """Grid holding local north coordinates,
            see :attr:`kite.scene.Frame.gridN`.
        :type: :class:`numpy.ndarray`
        """
        return self.scene.frame.gridN[self._slice_rows, self._slice_cols]

    @property_cached
    def gridNmeter(self):
        """Grid holding local north coordinates,
            see :attr:`kite.scene.Frame.gridNmeter`.
        :type: :class:`numpy.ndarray`
        """
        return self.scene.frame.gridNmeter[self._slice_rows, self._slice_cols]

    @property
    def llE(self):
        """
        :getter: Lower left east coordinate in local coordinates
            (*meters* or *degree*).
        :type: float
        """
        return self.scene.frame.E[self.llc]

    @property
    def llN(self):
        """
        :getter: Lower left north coordinate in local coordinates
            (*meter* or *degree*).
        :type: float
        """
        return self.scene.frame.N[self.llr]

    @property
    def urN(self):
        return self.llN + self.sizeN

    @property
    def urE(self):
        return self.llE + self.sizeE

    @property_cached
    def sizeE(self):
        """
        :getter: Size in eastern direction in *meters* or *degree*.
        :type: float
        """
        sizeE = self.length * self.scene.frame.dE
        if (self.llE + sizeE) > self.scene.frame.E.max():
            sizeE = self.scene.frame.E.max() - self.llE
        return sizeE

    @property_cached
    def sizeN(self):
        """
        :getter: Size in northern direction in *meters* or *degree*.
        :type: float
        """
        sizeN = self.length * self.scene.frame.dN
        if (self.llN + sizeN) > self.scene.frame.N.max():
            sizeN = self.scene.frame.N.max() - self.llN
        return sizeN

    def iterChildren(self):
        """Iterator over the all children.

        :yields: Children of it's own.
        :type: :class:`~kite.quadtree.QuadNode`
        """
        yield self
        if self.children is not None:
            for c in self.children:
                yield from c.iterChildren()

    def iterLeaves(self):
        """Iterator over the leaves, evaluating parameters from
        :class:`~kite.Quadtree` instance.

        :yields: Leafs fulfilling the tree's parameters.
        :type: :class:`~kite.quadtree.QuadNode`
        """
        if (
            (
                self.quadtree._corr_func(self) < self.quadtree.epsilon
                and not self.length > self.quadtree._tile_size_lim_px[1]
            )
            or self.children is None
            or (self.children[0].length < self.quadtree._tile_size_lim_px[0])
        ):
            yield self
        else:
            for c in self.children:
                yield from c.iterLeaves()

    def _iterSplitNode(self):
        if self.length == 1:
            yield None
        for nr, nc in self.CORNERS:
            n = QuadNode(
                self.quadtree,
                self._displacement,
                self.llr + self.length / 2 * nr,
                self.llc + self.length / 2 * nc,
                self.length / 2,
            )
            if n.displacement.size == 0 or np.all(n.displacement_mask):
                continue
            yield n

    def createTree(self):
        """Create the tree from a set of basenodes, ignited by
        :class:`~kite.Quadtree` instance. Evaluates :class:`~kite.Quadtree`
        correction method and :attr:`~kite.Quadtree.epsilon_min`.
        """
        if (
            self.quadtree._corr_func(self) > self.quadtree.epsilon_min
            or self.length >= 64
        ) and not self.length < self.MIN_PIXEL_LENGTH_NODE:
            # self.length > .1 * max(self.quadtree._data.shape): !! Expensive
            self.children = tuple(c for c in self._iterSplitNode())
            if len(self.children) == 0:
                self.children = None
            else:
                for c in self.children:
                    c.createTree()
        else:
            self.children = None


class QuadtreeConfig(guts.Object):
    """Quadtree configuration object holding essential parameters used to
    reconstruct a particular tree
    """

    correction = guts.StringChoice.T(
        choices=("mean", "median", "bilinear", "std"),
        default="median",
        help="Node correction for splitting, available methods "
        " ``['mean', 'median', 'bilinear', 'std']``",
    )
    epsilon = guts.Float.T(
        optional=True, help="Variance threshold when a node is split"
    )
    nan_allowed = guts.Float.T(default=0.9, help="Allowed NaN fraction per tile")
    tile_size_min = guts.Float.T(
        optional=True, help="Minimum allowed tile size in *meters* or *degree*"
    )
    tile_size_max = guts.Float.T(
        optional=True, help="Maximum allowed tile size in *meters* or *degree*"
    )
    leaf_blacklist = guts.List.T(
        optional=True, default=[], help="Blacklist of excluded leaves"
    )


class Quadtree(object):
    """Quadtree for irregular subsampling InSAR displacement data held in
    :py:class:`kite.scene.Scene`

    InSAR displacement scenes can hold a vast amount of data points,
    which is often highly redundant and unsuitably large for the use in
    inverse modeling. By subsampling and therefore decimating the data points
    systematically through a parametrized quadtree we can reduce the dataset
    without significant loss of displacement information. Quadtree subsampling
    keeps a high spatial resolution where displacement gradients are high and
    efficiently reduces data point density in regions with small displacement
    variations. The product is a manageable dataset size with good
    representation of the original data.

    The standard deviation from :attr:`kite.quadtree.QuadNode.displacement`
    is evaluated against different corrections:

        * ``mean``: Mean is subtracted
        * ``median``: Median is subtracted
        * ``bilinear``: A 2D detrend is applied to the node
        * ``std``:  Pure standard deviation without correction

    set through :func:`~kite.Quadtree.setCorrection`. If the standard deviation
    exceeds :attr:`~kite.Quadtree.epsilon` the node is split.

    The leaves can also be exported in a *CSV* format by
    :func:`~kite.Quadtree.export_csv`, or *GeoJSON* by
    :func:`~kite.Quadtree.export_geojson`.

    Controlling attributes are:

        * :attr:`~kite.Quadtree.epsilon`, RMS threshold
        * :attr:`~kite.Quadtree.nan_fraction`, allowed :attr:`numpy.nan` in
          node
        * :attr:`~kite.Quadtree.tile_size_max`, maximum node size in
            *meters* or *degree*
        * :attr:`~kite.Quadtree.tile_size_min`, minimum node size in
            *meter* or *degree*

    :attr:`~kite.Quadtree.leaves` hold the current tree's
    :class:`~kite.quadtree.QuadNode` 's.
    """

    _displacement_corrections = {
        "mean": ("Standard deviation around mean", lambda n: n.corr_mean),
        "median": ("Standard deviation around median", lambda n: n.corr_median),
        "bilinear": (
            "Standard deviation around bilinear detrended node",
            lambda n: n.corr_bilinear,
        ),
        "std": ("Standard deviation (std)", lambda n: n.std),
    }

    _norm_methods = {
        "mean": lambda n: n.mean,
        "median": lambda n: n.median,
        "weight": lambda n: n.weight,
    }

    def __init__(self, scene, config=None):
        self.evChanged = Subject()
        self.evConfigChanged = Subject()
        self._leaves = None
        self.scene = scene
        self.displacement = self.scene.displacement
        self.frame = self.scene.frame
        self._scene_state = None

        # Cached matrices
        self._leaf_matrix_means = np.empty_like(self.displacement)
        self._leaf_matrix_medians = np.empty_like(self.displacement)
        self._leaf_matrix_weights = np.empty_like(self.displacement)

        self._log = scene._log.getChild("Quadtree")
        self.setConfig(config or QuadtreeConfig())

        self.scene.evConfigChanged.subscribe(self.setConfig)
        # self.scene.evChanged.subscribe(self.reinitializeTree)

    def setConfig(self, config=None):
        """Sets and updated the config of the instance

        :param config: New config instance, defaults to configuration provided
                       by parent :class:`~kite.Scene`
        :type config: :class:`~kite.covariance.QuadtreeConfig`, optional
        """
        if config is None:
            config = self.scene.config.quadtree

        if self.scene.config.old_import:
            frame = self.scene.config.frame

            from pyrocko import orthodrome as od

            self._log.warning("Old format - converting quadtree configuration")

            dLat, dLon = od.ne_to_latlon(
                frame.llLat, frame.llLon, config.tile_size_max, config.tile_size_min
            )

            config.tile_size_min = dLon - frame.llLon
            config.tile_size_max = dLat - frame.llLat

        self.config = config
        self.setCorrection(self.config.correction)

        self.evConfigChanged.notify()

    def setCorrection(self, correction="mean"):
        """Set correction method calculating the standard deviation of
        instances :class:`~kite.quadtree.QuadNode` s

        The standard deviation from :attr:`kite.quadtree.QuadNode.displacement`
        is evaluated against different corrections:

        * ``mean``: Mean is subtracted
        * ``median``: Median is subtracted
        * ``bilinear``: A 2D detrend is applied to the node
        * ``std``:  Pure standard deviation without correction

        :param correction: Choose from methods
            ``mean_std, median_std, bilinear_std, std``
        :type correction: str
        :raises: AttributeError
        """
        if correction not in self._displacement_corrections.keys():
            raise AttributeError(
                "Method %s not in %s", correction, self._displacement_corrections
            )
        self._log.debug("Changing to split method '%s'", correction)

        self.config.correction = correction
        self._corr_func = self._displacement_corrections[correction][1]
        self.reinitializeTree()

    def ensureTree(self):
        if self._scene_state != self.scene.get_plugin_state_hash():
            self.reinitializeTree()

    def reinitializeTree(self):
        # Clearing cached properties through None
        self.leaf_center_distance = None
        self.nodes = None
        self.epsilon_min = None
        self._epsilon_init = None
        self.clearLeaves()
        self.epsilon = self.config.epsilon or self._epsilon_init

        self._initTree()
        if self.nleaves == 0:
            self._log.warning(
                "No leaves in default quadtree," " setting allowed_nan=1."
            )
            self.nan_allowed = 1.0

        self.evChanged.notify()

    def clearLeaves(self):
        """Clear cached leafs and properties"""
        self.leaves = None
        self.leaf_center_distance = None
        self.leaf_los_rotation_factors = None
        self.leaf_means = None
        self.leaf_medians = None

    @property
    def min_node_length_px(self):
        npx = max(self.frame.cols, self.frame.rows)
        return int(2 ** round(np.log(npx / 64)))

    def _initTree(self):
        QuadNode.MIN_PIXEL_LENGTH_NODE = (
            QuadNode.MIN_PIXEL_LENGTH_NODE or self.min_node_length_px
        )

        t0 = time.time()
        for b in self._base_nodes:
            b.createTree()

        self._scene_state = self.scene.get_plugin_state_hash()
        self._log.debug(
            "Tree created, %d nodes [%0.4f s]", self.nnodes, time.time() - t0
        )

    @property
    def epsilon(self):
        """Threshold for quadtree splitting its ``QuadNode``.

        The threshold is the maximum standard deviation of leaf mean,
        median or simply its values (see ''SetSplitMethod'') allowed to
        not further split a "QuadNode".

        :setter: Sets the epsilon/RMS threshold
        :getter: Returns the current epsilon
        :type: float
        """
        return self.config.epsilon

    @epsilon.setter
    def epsilon(self, value):
        value = float(value)
        if self.config.epsilon == value:
            return
        if value < self.epsilon_min:
            self._log.warning(
                "Epsilon is out of bounds [%0.6f], epsilon_min %0.6f",
                value,
                self.epsilon_min,
            )
            return
        self.clearLeaves()
        self.clearLeafBlacklist()
        self.config.epsilon = value

        self.evChanged.notify()

    @property_cached
    def _epsilon_init(self):
        """Initial epsilon for virgin tree creation"""
        return np.nanstd(self.displacement)

    @property_cached
    def epsilon_min(self):
        """Lowest allowed epsilon
        :type: float
        """
        return self._epsilon_init * 0.1

    @property
    def nan_allowed(self):
        """Fraction of allowed ``NaN`` values in quadtree leaves. If
        value is exceeded the leaf is kicked out entirely.

        :setter: Fraction  ``0. <= fraction <= 1``.
        :type: float
        """
        return self.config.nan_allowed

    @nan_allowed.setter
    def nan_allowed(self, value):
        if value > 1.0 or value <= 0.0:
            self._log.warning("NaN fraction must be 0. < nan_allowed <= 1.")
            return

        self.clearLeaves()
        self.clearLeafBlacklist()
        self.config.nan_allowed = value
        self.evChanged.notify()

    @property
    def tile_size_min(self):
        """Minimum allowed tile size in *meter*.
        Measured along long edge ``(max(dE, dN))``.
        Minimum tile size defaults to 1/20th of the largest dimension

        :getter: Returns the minimum allowed tile size
        :setter: Sets the minimum threshold
        :type: float
        """
        if self.config.tile_size_min is None:
            frame = self.scene.frame
            max_px = max(frame.shape)
            self.config.tile_size_min = max(frame.dE, frame.dN) * (max_px / 20)

        return self.config.tile_size_min

    @tile_size_min.setter
    def tile_size_min(self, value):
        if value > self.tile_size_max:
            self._log.warning("tile_size_min > tile_size_max is required!")
            return
        self.config.tile_size_min = value
        self._tileSizeChanged()

    @property
    def tile_size_max(self):
        """Maximum allowed tile size in *meter*.
        Measured along long edge ``(max(dE, dN))``
        Maximum tile size defaults to 1/5th of the largest dimension

        :getter: Returns the maximum allowed tile size
        :setter: Sets the maximum threshold
        :type: float
        """
        if self.config.tile_size_max is None:
            frame = self.scene.frame
            max_px = max(frame.shape)
            self.config.tile_size_max = max(frame.dE, frame.dN) * (max_px / 5)

        return self.config.tile_size_max

    @tile_size_max.setter
    def tile_size_max(self, value):
        if value < self.tile_size_min:
            self._log.warning("tile_size_min > tile_size_max is required")
            return
        self.config.tile_size_max = value
        self._tileSizeChanged()

    def _tileSizeChanged(self):
        self._tile_size_lim_px = None
        self.clearLeaves()
        self.clearLeafBlacklist()
        self.evChanged.notify()

    @property_cached
    def _tile_size_lim_px(self):
        dpx = max(self.scene.frame.dE, self.scene.frame.dN)
        return (round(self.tile_size_min / dpx), round(self.tile_size_max / dpx))

    @property_cached
    def nodes(self):
        """All nodes of the tree

        :getter: Get the list of nodes
        :type: list
        """
        return [n for b in self._base_nodes for n in b.iterChildren()]

    @property
    def nnodes(self):
        """
        :getter: Number of nodes of the built tree.
        :type: int
        """
        return len(self.nodes)

    def clearLeafBlacklist(self):
        self.config.leaf_blacklist = []

    def blacklistLeaves(self, leaves):
        """Blacklist a leaf and exclude it from the tree

        :param leaves: Leaf instances
        :type leaves: list
        """
        self.config.leaf_blacklist.extend(leaves)
        self._log.debug(
            "Blacklisted leaves: %s" % ", ".join(self.config.leaf_blacklist)
        )
        self.clearLeaves()
        self.evChanged.notify()

    @property_cached
    def leaves(self):
        """:getter: List of leaves for current configuration.
        :type: (list or :class:`~kite.quadtree.QuadNode` s)
        """
        t0 = time.time()
        leaves = []
        for b in self._base_nodes:
            leaves.extend(
                [
                    lf
                    for lf in b.iterLeaves()
                    if lf.nan_fraction < self.nan_allowed
                    and lf.id not in self.config.leaf_blacklist
                ]
            )
        self._log.debug(
            "Gathering leaves for epsilon %.4f (nleaves=%d) [%0.4f s]"
            % (self.epsilon, len(leaves), time.time() - t0)
        )
        return leaves

    @property
    def nleaves(self):
        """
        :getter: Number of leaves for current parametrisation.
        :type: int
        """
        return len(self.leaves)

    @property
    def leaf_mean_px_var(self):
        """
        :getter: Mean pixel variance in each quadtree,
            if :attr:`kite.Scene.displacement_px_var` is set.
        :type: :class:`numpy.ndarray`, size ``N``.
        """
        if self.scene.displacement_px_var is not None:
            return np.array([lf.mean_px_var for lf in self.leaves])
        return None

    @property_cached
    def leaf_means(self):
        """
        :getter: Leaf mean displacements from
            :attr:`kite.quadtree.QuadNode.mean`.
        :type: :class:`numpy.ndarray`, size ``N``.
        """
        return np.array([lf.mean for lf in self.leaves])

    @property_cached
    def leaf_medians(self):
        """
        :getter: Leaf median displacements from
            :attr:`kite.quadtree.QuadNode.median`.
        :type: :class:`numpy.ndarray`, size ``N``.
        """
        return np.array([lf.median for lf in self.leaves])

    @property
    def _leaf_focal_points(self):
        return np.array([lf._focal_point for lf in self.leaves])

    @property
    def leaf_focal_points(self):
        """
        :getter: Leaf focal points in local coordinates.
        :type: :class:`numpy.ndarray`, size ``(N, 2)``
        """
        return np.array([lf.focal_point for lf in self.leaves])

    @property
    def leaf_focal_points_meter(self):
        """
        :getter: Leaf focal points in meter.
        :type: :class:`numpy.ndarray`, size ``(N, 2)``
        """
        return np.array([lf.focal_point_meter for lf in self.leaves])

    @property
    def leaf_coordinates(self):
        """Synonym for :func:`Quadtree.leaf_focal_points`
        in easting/northing"""
        return self.leaf_focal_points

    @property_cached
    def leaf_center_distance(self):
        """
        :getter: Leaf distance to center point of the quadtree
        :type: :class:`numpy.ndarray`, size ``(N, 3)``
        """
        distances = np.empty((self.nleaves, 3))
        center = self.center_point
        distances[:, 0] = self.leaf_focal_points[:, 0] - center[0]
        distances[:, 1] = self.leaf_focal_points[:, 1] - center[1]
        distances[:, 2] = np.sqrt(distances[:, 1] ** 2 + distances[:, 1] ** 2)
        return distances

    @property
    def leaf_eastings(self):
        return self.leaf_coordinates[:, 0]

    @property
    def leaf_northings(self):
        return self.leaf_coordinates[:, 1]

    @property
    def leaf_phis(self):
        """
        :getter: Median leaf LOS phi angle. :attr:`kite.Scene.phi`
        :type: :class:`numpy.ndarray`, size ``(N)``
        """
        return np.array([lf.phi for lf in self.leaves])

    @property
    def leaf_thetas(self):
        """
        :getter: Median leaf LOS theta angle. :attr:`kite.Scene.theta`
        :type: :class:`numpy.ndarray`, size ``(N)``
        """
        return np.array([lf.theta for lf in self.leaves])

    @property_cached
    def leaf_los_rotation_factors(self):
        """
        :getter: Trigonometric factors for rotating displacement
            matrices towards LOS.
            See :attr:`kite.BaseScene.los_rotation_factors`
        :type: :class:`numpy.ndarray`, Nx3
        """
        los_factors = np.empty((self.nleaves, 3))
        los_factors[:, 0] = np.sin(self.leaf_thetas)
        los_factors[:, 1] = np.cos(self.leaf_thetas) * np.cos(self.leaf_phis)
        los_factors[:, 2] = np.cos(self.leaf_thetas) * np.sin(self.leaf_phis)
        return los_factors

    @property
    def leaf_matrix_means(self):
        """
        :getter: Leaf mean displacements casted to
            :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N, M)``
        """
        return self._getLeafsNormMatrix(self._leaf_matrix_means, method="mean")

    @property
    def leaf_matrix_medians(self):
        """
        :getter: Leaf median displacements casted to
            :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N, M)``
        """
        return self._getLeafsNormMatrix(self._leaf_matrix_medians, method="median")

    @property
    def leaf_matrix_weights(self):
        """
        :getter: Leaf weights casted to :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N, M)``
        """
        return self._getLeafsNormMatrix(self._leaf_matrix_weights, method="weight")

    def _getLeafsNormMatrix(self, array, method="median"):
        if method not in self._norm_methods.keys():
            raise AttributeError(
                "Method %s is not in %s" % (method, list(self._norm_methods.keys()))
            )

        array.fill(np.nan)
        for lf in self.leaves:
            array[lf._slice_rows, lf._slice_cols] = self._norm_methods[method](lf)
        array[self.scene.displacement_mask] = np.nan
        return array

    @property
    def center_point(self):
        return np.median(self.leaf_focal_points, axis=0)

    @property
    def reduction_efficiency(self):
        """This is measure for the reduction of the scene's full resolution
        over the quadtree.

        :getter: Quadtree efficiency as :math:`N_{full} / N_{leaves}`
        :type: float
        """
        return (self.scene.rows * self.scene.cols) / (
            self.nleaves if self.nleaves else 1
        )

    @property
    def reduction_rms(self):
        """The RMS error is defined between
        :attr:`~kite.Quadtree.leaf_matrix_means` and
        :attr:`kite.Scene.displacement`.

        :getter: The reduction RMS error
        :type: float
        """
        if np.all(np.isnan(self.leaf_matrix_means)):
            return np.inf
        return np.sqrt(
            np.nanmean((self.scene.displacement - self.leaf_matrix_means) ** 2)
        )

    @property_cached
    def _base_nodes(self):
        self._base_nodes = []
        init_length = np.power(
            2, np.ceil(np.log(np.min(self.displacement.shape)) / np.log(2))
        )
        nx, ny = np.ceil(np.array(self.displacement.shape) / init_length)
        self._log.debug("Creating %d base nodes", nx * ny)

        displacement = self.scene.displacement
        for ir in range(int(nx)):
            for ic in range(int(ny)):
                llr = ir * init_length
                llc = ic * init_length
                node = QuadNode(self, displacement, llr, llc, init_length)
                self._base_nodes.append(node)

        if len(self._base_nodes) == 0:
            raise AssertionError("Could not init base nodes.")
        return self._base_nodes

    @property_cached
    def plot(self):
        """Simple `matplotlib` illustration of the quadtree

        :type: :attr:`Quadtree.leaf_matrix_means`.
        """
        from kite.plot2d import QuadtreePlot

        return QuadtreePlot(self)

    def getStaticTarget(self):
        """Not Implemented"""
        raise NotImplementedError

    def getMPLRectangles(self):
        """
        Get the quadtree as a list of matplotlib rectangles.

        :returns: Rectangles for plotting
        :rtype: list of :class:`matplotlib.patcjes.Rectangle`
        """
        from matplotlib.patches import Rectangle

        rectangles = []
        for lf in self.leaves:
            r = Rectangle((lf.llE, lf.llN), lf.sizeE, lf.sizeN)
            rectangles.append(r)
        return rectangles

    def export_csv(self, filename):
        """ Exports the current quadtree leaves to ``filename`` in a
        *CSV* format

        The formatting is::

            # node_id, focal_point_E, focal_point_N, theta, phi, \
            mean_displacement, median_displacement, absolute_weight

        :param filename: export_csv to path
        :type filename: string
        """
        self._log.debug("Exporting Quadtree as to %s", filename)
        with open(filename, mode="w") as f:
            f.write(
                "# node_id, focal_point_E, focal_point_N, theta, phi,"
                " unitE, unitN, unitU,"
                " mean_displacement, median_displacement, absolute_weight\n"
            )
            for lf in self.leaves:
                f.write(
                    "{lf.id}, {lf.focal_point[0]}, {lf.focal_point[1]}, "
                    "{lf.theta}, {lf.phi}, {lf.unitE}, {lf.unitN}, {lf.unitU},"
                    " {lf.mean}, {lf.median}, {lf.weight}\n".format(lf=lf)
                )

    def export_geojson(self, filename):
        import geojson

        self._log.debug("Exporting GeoJSON Quadtree to %s", filename)
        features = []

        for lf in self.leaves:
            llN, llE, urN, urE = (lf.llN, lf.llE, lf.urN, lf.urE)

            if self.frame.isDegree():
                llN += self.frame.llLat
                llE += self.frame.llLon
                urN += self.frame.llLat
                urE += self.frame.llLon

            coords = np.array(
                [(llN, llE), (llN, urE), (urN, urE), (urN, llE), (llN, llE)]
            )

            if self.frame.isMeter():
                coords = od.ne_to_latlon(self.frame.llLat, self.frame.llLon, *coords.T)
                coords = np.array(coords).T

            coords = coords[:, [1, 0]].tolist()

            feature = geojson.Feature(
                geometry=geojson.Polygon(coordinates=[coords]),
                id=lf.id,
                properties={
                    "mean": float(lf.mean),
                    "median": float(lf.median),
                    "std": float(lf.std),
                    "var": float(lf.var),
                    "phi": float(lf.phi),
                    "theta": float(lf.theta),
                    "unitE": float(lf.unitE),
                    "unitN": float(lf.unitN),
                    "unitU": float(lf.unitU),
                },
            )
            features.append(feature)

        collection = geojson.FeatureCollection(features)
        with open(filename, "w") as f:
            geojson.dump(collection, f)

    def get_state_hash(self):
        sha = sha1()
        sha.update(str(self.config).encode())
        return sha.digest().hex()


__all__ = ["Quadtree", "QuadtreeConfig"]


if __name__ == "__main__":
    from kite.scene import SceneSynTest

    sc = SceneSynTest.createGauss(2000, 2000)

    for e in np.linspace(0.1, 0.00005, num=30):
        sc.quadtree.epsilon = e
    # qp = Plot2DQuadTree(qt, cmap='spectral')
    # qp.plot()
