#!/usr/bin/python
# -*- coding: utf-8 -*-
from hashlib import sha1

import numpy as np
import scipy as sp

try:
    from scipy import fftpack as fft
except ImportError:
    from scipy import fft

import time

from pyrocko import guts
from pyrocko.guts_array import Array

from kite import covariance_ext
from kite.util import (
    Subject,
    derampMatrix,
    property_cached,  # noqa
    trimMatrix,
)

__all__ = ["Covariance", "CovarianceConfig"]

NOISE_PATCH_MIN_PX = 256 * 256
NOISE_PATCH_MAX_NAN = 0.8

noise_regimes = [
    (1.0 / 2000, np.inf),
    (1.0 / 2000, 1.0 / 500),
    (1.0 / 500, 1.0 / 10),
    (0, np.inf),
]


def modelCovarianceExponential(distance, a, b):
    """Exponential function model to approximate a positive-definite covariance

    We assume the following simple covariance model to describe the empirical
    noise observations:

    .. math::

        cov(d) = c \\cdot e^{\\frac{-d}{b}}

    :param distance: Distance between
    :type distance: float or :class:`numpy.ndarray`
    :param a: Linear model coefficient
    :type a: float
    :param b: Exponential model coefficient
    :type b: float
    :returns: Covariance at ``distance``
    :rtype: :class:`numpy.ndarray`
    """
    return a * np.exp(-distance / b)


def modelCovarianceExponentialCosine(distance, a, b, c, d):
    r"""Exponential function model to approximate a positive-definite covariance

    We assume the following simple covariance model to describe the empirical
    noise observations:

    .. math::

        cov(d) = c \\cdot e^{\\frac{-d}{b}} \\cdot \cos{\\frac{d-c}{d}}

    :param distance: Distance between
    :type distance: float or :class:`numpy.ndarray`
    :param a: Linear model coefficient
    :type a: float
    :param b: Exponential model coefficient
    :type b: float
    :param c: Cosinus distance correction
    :type c: float
    :param c: Cosinus coefficient
    :type c: float
    :returns: Covariance at ``distance``
    :rtype: :class:`numpy.ndarray`
    """
    return a * np.exp(-distance / b) * np.cos((distance - c) / d)


def modelPowerspec(k, beta, D):
    """Exponential linear model to estimate a log-linear power spectrum

    We assume the following log-linear model for a measured power spectrum:

    .. math::

        pow(k) = \\frac{k^\\beta}{D}


    :param k: Wavenumber
    :type k: float or :class:`numpy.ndarray`
    :param a: Exponential model factor
    :type a: float
    :param b: Fractional model factor
    :type b: float
    """
    return (k**beta) / D


class CovarianceConfig(guts.Object):
    noise_coord = Array.T(
        shape=(None,),
        dtype=float,
        serialize_as="list",
        optional=True,
        help="Noise patch coordinates and size,",
    )
    model_coefficients = guts.Tuple.T(
        optional=True,
        help="Covariance model coefficients. Either two (exponential) "
        "or three (exponential and cosine term) coefficients."
        "See also :func:`~kite.covariance.modelCovariance`.",
    )
    model_function = guts.StringChoice.T(
        choices=["exponential", "exponential_cosine"],
        default="exponential",
        help="Covariance approximation function.",
    )
    sampling_method = guts.StringChoice.T(
        choices=["spectral", "spatial"],
        default="spatial",
        help="Method for estimating the covariance and structure function.",
    )
    spatial_bins = guts.Int.T(
        default=75, help="Number of distance bins for spatial covariance sampling."
    )
    spatial_pairs = guts.Int.T(
        default=200000, help="Number of random pairs for spatial covariance sampling."
    )
    variance = guts.Float.T(optional=True, help="Variance of the model.")
    adaptive_subsampling = guts.Bool.T(
        default=True, help="Adaptive subsampling flag for full covariance calculation."
    )
    covariance_matrix = Array.T(
        dtype=float,
        optional=True,
        serialize_as="base64",
        help="Cached covariance matrix, "
        "see :attr:`~kite.Covariance.covariance_matrix`.",
    )

    def __init__(self, *args, **kwargs):
        if len(kwargs) != 0:
            if "a" in kwargs and "b" in kwargs:
                kwargs["model_coefficients"] = (kwargs.pop("a"), kwargs.pop("b"))
        guts.Object.__init__(self, *args, **kwargs)


class Covariance(object):
    """Construct the variance-covariance matrix of quadtree subsampled data.

    Variance and covariance estimates are used to construct the weighting
    matrix to be used later in an optimization.

    Two different methods exist to propagate full-resolution data variances
    and covariances of :class:`kite.Scene.displacement` to the
    covariance matrix of the subsampled dataset:

    1. The distance between :py:class:`kite.quadtree.QuadNode`
       leaf focal points, :py:class:`kite.covariance.Covariance.matrix_focal`
       defines the approximate covariance of the quadtree leaf pair.
    2. The _accurate_ propagation of covariances by taking the mean of
       every node pair pixel covariances. This process is computational
       very expensive and can take a few minutes.
       :py:class:`kite.covariance.Covariance.matrix_focal`

    :param quadtree: Quadtree to work on
    :type quadtree: :class:`~kite.Quadtree`
    :param config: Config object
    :type config: :class:`~kite.covariance.CovarianceConfig`
    """

    def __init__(self, scene, config=CovarianceConfig()):
        self.evChanged = Subject()
        self.evConfigChanged = Subject()

        self.frame = scene.frame
        self.quadtree = scene.quadtree
        self.scene = scene
        self.nthreads = 0
        self._noise_data = None
        self._powerspec1d_cached = None
        self._powerspec2d_cached = None
        self._powerspec3d_cached = None
        self._noise_data_grid = None
        self._initialized = False
        self._log = scene._log.getChild("Covariance")

        self.setConfig(config)
        self.quadtree.evChanged.subscribe(self._clear)
        self.scene.evConfigChanged.subscribe(self.setConfig)

    def __call__(self, *args, **kwargs):
        return self.getLeafCovariance(*args, **kwargs)

    def setConfig(self, config=None):
        """Sets and updated the config of the instance

        :param config: New config instance, defaults to configuration provided
                       by parent :class:`~kite.Scene`
        :type config: :class:`~kite.covariance.CovarianceConfig`, optional
        """
        if config is None:
            config = self.scene.config.covariance

        if self.scene.config.old_import:
            self._log.warning("Old format - resetting noise patch coordinates")
            config.covariance_matrix = None
            config.noise_coord = None

        self.config = config
        if config.noise_coord is None and (
            config.model_coefficients is not None or config.variance is not None
        ):
            self.noise_data  # init data array
            self.config.model_coefficients = config.model_coefficients
            self.config.variance = config.variance

        self._clear(config=False)
        self.evConfigChanged.notify()

    def _clear(self, config=True, spectrum=True):
        if config:
            self.config.model_coefficients = None
            self.config.variance = None
            self.config.covariance_matrix = None

        if spectrum:
            self.structure_spectral = None
            self._powerspec1d_cached = None
            self._powerspec2d_cached = None

        self._noise_data_grid = None
        self.covariance_matrix = None
        self.covariance_matrix_focal = None
        self.covariance_spectral = None
        self.covariance_spatial = None
        self.structure_spatial = None
        self.weight_matrix = None
        self.weight_matrix_focal = None
        self._initialized = False
        self.evChanged.notify()

    @property
    def finished_combinations(self):
        return covariance_ext.get_finished_combinations()

    @property
    def noise_coord(self):
        """Coordinates of the noise patch in local coordinates.

        :setter: Set the noise coordinates
        :getter: Get the noise coordinates
        :type: :class:`numpy.ndarray`, ``[llE, llN, sizeE, sizeN]``
        """
        if self.config.noise_coord is None:
            self.noise_data
        return self.config.noise_coord

    @noise_coord.setter
    def noise_coord(self, values):
        self.config.noise_coord = np.array(values)

    @property
    def noise_patch_size_km2(self):
        """
        :getter: Noise patch size in :math:`km^2`.
        :type: float
        """
        if self.noise_coord is None:
            return 0.0
        size = (self.noise_coord[2] * self.noise_coord[3]) * 1e-6
        if self.noise_data.size < self.NOISE_PATCH_MIN_PX:
            self._log.warning("Defined noise patch is instably small")
        return size

    @property
    def noise_data(self, data):
        """Noise data we process to estimate the covariance

        :setter: Set the noise patch to analyze the covariance.
        :getter: If the noise data has not been set manually, we grab data
                 through :func:`~kite.Covariance.selectNoiseNode`.
        :type: :class:`numpy.ndarray`
        """
        return self._noise_data

    @noise_data.getter
    def noise_data(self):
        if self._noise_data is not None:
            return self._noise_data
        elif self.config.noise_coord is not None:
            self._log.debug("Selecting noise_data from config...")
            llE, llN = self.scene.frame.mapENMatrix(*self.config.noise_coord[:2])
            sE, sN = self.scene.frame.mapENMatrix(*self.config.noise_coord[2:])
            slice_E = slice(llE, llE + sE)
            slice_N = slice(llN, llN + sN)

            covariance_matrix = self.config.covariance_matrix
            self.noise_data = self.scene.displacement[slice_N, slice_E]
            self.config.covariance_matrix = covariance_matrix
        else:
            self._log.debug("Selecting noise_data from Quadtree...")
            node = self.selectNoiseNode()
            self.noise_data = node.displacement
            self.noise_coord = [node.llE, node.llN, node.sizeE, node.sizeN]

        return self._noise_data

    @noise_data.setter
    def noise_data(self, data):
        data = data.copy()
        data = derampMatrix(trimMatrix(data))
        data[np.isnan(data)] = 0.0
        self._noise_data = data
        self._clear()

    @property
    def noise_data_gridE(self):
        return self._get_noise_data_grid()[0]

    @property
    def noise_data_gridN(self):
        return self._get_noise_data_grid()[1]

    def _get_noise_data_grid(self):
        if self._noise_data_grid is None:
            scene = self.scene

            llE, llN = scene.frame.mapENMatrix(*self.noise_coord[:2])
            sE, sN = scene.frame.mapENMatrix(*self.noise_coord[2:])
            slice_E = slice(llE, llE + sE + 1)
            slice_N = slice(llN, llN + sN + 1)

            gridE = scene.frame.gridEmeter[slice_N, slice_E]
            gridN = scene.frame.gridNmeter[slice_N, slice_E]

            gridE = trimMatrix(self.noise_data, data=gridE)
            gridN = trimMatrix(self.noise_data, data=gridN)

            self._noise_data_grid = (gridE, gridN)

        return self._noise_data_grid

    def selectNoiseNode(self):
        """Choose noise node from quadtree
        the biggest :class:`~kite.quadtree.QuadNode` from
        :class:`~kite.Quadtree`.

        :returns: A quadnode with the least signal.
        :rtype: :class:`~kite.quadtree.QuadNode`
        """
        t0 = time.time()

        node_selection = [
            n
            for n in self.quadtree.nodes
            if n.npixel > NOISE_PATCH_MIN_PX and n.nan_fraction < NOISE_PATCH_MAX_NAN
        ]
        if not node_selection:
            node_selection = self.quadtree.leaves

        stdmax = max([n.std for n in node_selection])
        lmax = max([n.std for n in node_selection])

        def costFunction(n):
            nl = np.log2(n.length) / np.log2(lmax)
            ns = n.std / stdmax
            return nl * (1.0 - ns) * (1.0 - n.nan_fraction)

        fitness = np.array([costFunction(n) for n in node_selection])

        self._log.debug(
            "Fetched noise from Quadtree.nodes [%0.4f s]" % (time.time() - t0)
        )
        node = node_selection[np.argmin(fitness)]
        return node

    def _mapLeaves(self, nx, ny):
        """Helper function returning appropriate
            :class:`~kite.quadtree.QuadNode` and for maintaining
            the internal mapping with the matrices.

        :param nx: matrix x position
        :type nx: int
        :param ny: matrix y position
        :type ny: int
        :returns: tuple of :class:`~kite.quadtree.QuadNode` s for ``nx``
            and ``ny``
        :rtype: tuple
        """
        leaf1 = self.quadtree.leaves[nx]
        leaf2 = self.quadtree.leaves[ny]

        self._leaf_mapping[leaf1.id] = nx
        self._leaf_mapping[leaf2.id] = ny

        return leaf1, leaf2

    def isFullCovarianceCalculated(self):
        if self.config.covariance_matrix is None:
            return False
        return True

    @property_cached
    def covariance_matrix(self):
        """Covariance matrix calculated from mean of all pixel pairs
            inside the node pairs (full and accurate propagation).

        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves` x
            :class:`~kite.Quadtree.nleaves`)
        """
        if not isinstance(self.config.covariance_matrix, np.ndarray):
            self.config.covariance_matrix = self._calcCovarianceMatrix(method="full")
            self.evChanged.notify()
        elif self.config.covariance_matrix.ndim == 1:
            try:
                nl = self.quadtree.nleaves
                self.config.covariance_matrix = self.config.covariance_matrix.reshape(
                    nl, nl
                )
            except ValueError:
                self.config.covariance_matrix = None
                return self.covariance_matrix
        return self.config.covariance_matrix

    @property_cached
    def covariance_matrix_focal(self):
        """Approximate Covariance matrix from quadtree leaf pair
            distance only. Fast, use for intermediate steps only and
            finally use approach :attr:`~kite.Covariance.covariance_matrix`.

        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves` x
            :class:`~kite.Quadtree.nleaves`)
        """
        return self._calcCovarianceMatrix(method="focal")

    @property_cached
    def weight_matrix(self):
        """Weight matrix from full covariance :math:`cov^{-1}`.

        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves` x
            :class:`~kite.Quadtree.nleaves`)
        """
        return np.linalg.inv(self.covariance_matrix)

    @property_cached
    def weight_matrix_L2(self):
        """Weight matrix from full covariance :math:`\\sqrt{cov^{-1}}`.

        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves` x
            :class:`~kite.Quadtree.nleaves`)
        """
        incov = np.linalg.inv(self.covariance_matrix)
        return sp.linalg.sqrtm(incov)

    @property_cached
    def weight_matrix_focal(self):
        """Approximated weight matrix from fast focal method
            :math:`cov_{focal}^{-1}`.

        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves` x
            :class:`~kite.Quadtree.nleaves`)
        """
        try:
            return np.linalg.inv(self.covariance_matrix_focal)
        except np.linalg.LinAlgError as e:
            self._log.exception(e)
            return np.eye(self.covariance_matrix_focal.shape[0])

    @property_cached
    def weight_vector(self):
        """Weight vector from full covariance :math:`cov^{-1}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves`)
        """
        return np.sum(self.weight_matrix, axis=1)

    @property_cached
    def weight_vector_focal(self):
        """Weight vector from fast focal method
            :math:`\\sqrt{cov_{focal}^{-1}}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleaves`)
        """
        return np.sum(self.weight_matrix_focal, axis=1)

    def _calcCovarianceMatrix(self, method="focal", nthreads=None):
        """Constructs the covariance matrix.

        :param method: Either ``focal`` point distances are used - this is
            quick but only an approximation.
            Or ``full``, where the full quadtree pixel distances matrices are
            calculated , defaults to ``focal``
        :type method: str, optional
        :returns: Covariance matrix
        :rtype: thon:numpy.ndarray
        """
        self._initialized = True
        nthreads = nthreads or self.nthreads

        nl = len(self.quadtree.leaves)
        self._leaf_mapping = {}

        t0 = time.time()

        if method == "focal":
            model = self.getModelFunction()

            coords = self.quadtree.leaf_focal_points_meter
            dist_matrix = np.sqrt(
                (coords[:, 0] - coords[:, 0, np.newaxis]) ** 2
                + (coords[:, 1] - coords[:, 1, np.newaxis]) ** 2
            )
            cov_matrix = model(dist_matrix, *self.covariance_model)

            # adding variance
            if self.variance < cov_matrix.max():
                variance = cov_matrix.max()
            else:
                variance = self.variance
            if self.quadtree.leaf_mean_px_var is not None:
                self._log.debug("Adding variance from scene.displacement_px_var")
                variance += self.quadtree.leaf_mean_px_var
            np.fill_diagonal(cov_matrix, variance)

            for nx, ny in np.nditer(np.triu_indices_from(dist_matrix)):
                self._mapLeaves(nx, ny)

        elif method == "full":
            leaf_map = np.empty((len(self.quadtree.leaves), 4), dtype=np.uint32)
            for nl, leaf in enumerate(self.quadtree.leaves):
                leaf, _ = self._mapLeaves(nl, nl)
                leaf_map[nl, 0], leaf_map[nl, 1] = (
                    leaf._slice_rows.start,
                    leaf._slice_rows.stop,
                )
                leaf_map[nl, 2], leaf_map[nl, 3] = (
                    leaf._slice_cols.start,
                    leaf._slice_cols.stop,
                )

            nleaves = self.quadtree.nleaves
            cov_matrix = covariance_ext.covariance_matrix(
                self.scene.frame.gridEmeter.filled(),
                self.scene.frame.gridNmeter.filled(),
                leaf_map,
                self.covariance_model,
                self.variance,
                nthreads,
                self.config.adaptive_subsampling,
            ).reshape(nleaves, nleaves)

            if self.quadtree.leaf_mean_px_var is not None:
                self._log.debug("Adding variance from scene.displacement_px_var")
                cov_matrix[
                    np.diag_indices_from(cov_matrix)
                ] += self.quadtree.leaf_mean_px_var

        else:
            raise TypeError("Covariance calculation %s method not defined!" % method)

        self._log.debug(
            "Created covariance matrix - %s mode [%0.4f s]" % (method, time.time() - t0)
        )
        return cov_matrix

    def isMatrixPosDefinite(self, full=False):
        self._log.debug("Checking whether matrix is positive-definite")
        if full:
            matrix = self.covariance_matrix
        else:
            matrix = self.covariance_matrix_focal

        try:
            chol_decomp = np.linalg.cholesky(matrix)
        except np.linalg.linalg.LinAlgError:
            pos_def = False
        else:
            pos_def = ~np.all(np.iscomplex(chol_decomp))
        finally:
            if not pos_def:
                self._log.warning("Covariance matrix is not positive definite!")
            return pos_def

    @staticmethod
    def _leafFocalDistance(leaf1, leaf2):
        return np.sqrt(
            (leaf1.focal_point[0] - leaf2.focal_point[0]) ** 2
            + (leaf1.focal_point[1] - leaf2.focal_point[1]) ** 2
        )

    def _leafMapping(self, leaf1, leaf2):
        if not isinstance(leaf1, str):
            leaf1 = leaf1.id
        if not isinstance(leaf2, str):
            leaf2 = leaf2.id
        if not self._initialized:
            self.covariance_matrix_focal
        try:
            return self._leaf_mapping[leaf1], self._leaf_mapping[leaf2]
        except KeyError as e:
            raise KeyError("Unknown quadtree leaf with id %s" % e)

    def getLeafCovariance(self, leaf1, leaf2):
        """Get the covariance between ``leaf1`` and ``leaf2`` from
            distances.

        :param leaf1: Leaf one
        :type leaf1: str of `leaf.id` or :class:`~kite.quadtree.QuadNode`
        :param leaf2: Leaf two
        :type leaf2: str of `leaf.id` or :class:`~kite.quadtree.QuadNode`
        :returns: Covariance between ``leaf1`` and ``leaf2``
        :rtype: float
        """
        return self.covariance_matrix[self._leafMapping(leaf1, leaf2)]

    def getLeafWeight(self, leaf, model="focal"):
        """Get the total weight of ``leaf``, which is the summation of
            all single pair weights of :attr:`kite.Covariance.weight_matrix`.

        .. math ::

            w_{x} = \\sum_i W_{x,i}

        :param model: ``Focal`` or ``full``, default ``focal``
        :type model: str
        :param leaf: A leaf from :class:`~kite.Quadtree`
        :type leaf: :class:`~kite.quadtree.QuadNode`

        :returns: Weight of the leaf
        :rtype: float
        """
        (nl, _) = self._leafMapping(leaf, leaf)
        weight_mat = self.weight_matrix_focal
        return np.mean(weight_mat, axis=0)[nl]

    def syntheticNoise(
        self, shape=(1024, 1024), dEdN=None, anisotropic=False, rstate=None
    ):
        """Create random synthetic noise from data noise power spectrum.

        This function uses the power spectrum of the data noise
        (:attr:`noise_data`) (:func:`powerspecNoise`) to create synthetic
        noise, e.g. to use it for data perturbation in optinmizations.
        The default sampling distances are taken from
        :attr:`kite.scene.Frame.dE` and :attr:`kite.scene.Frame.dN`. They can
        be overwritten.

        :param shape: shape of the desired noise patch.
            Pixels in northing and easting (`nE`, `nN`),
            defaults to `(1024, 1024)`.
        :type shape: tuple, optional
        :param dEdN: The sampling distance in east and north [m], defaults to
            (:attr:`kite.scene.Frame.dEmeter`,
             :attr:`kite.scene.Frame.dNmeter`).
        :type dEdN: tuple, floats
        :returns: synthetic noise patch
        :rtype: :class:`numpy.ndarray`
        """
        if (shape[0] + shape[1]) % 2 != 0:
            # self._log.warning('Patch dimensions must be even, '
            #                   'ceiling dimensions!')
            pass
        nE = shape[1] + (shape[1] % 2)
        nN = shape[0] + (shape[0] % 2)

        if rstate is None:
            rstate = np.random.RandomState()

        rfield = rstate.rand(nN, nE)
        spec = np.fft.fft2(rfield)

        if not dEdN:
            dE, dN = (self.scene.frame.dEmeter, self.scene.frame.dNmeter)
        kE = np.fft.fftfreq(nE, dE)
        kN = np.fft.fftfreq(nN, dN)
        k_rad = np.sqrt(kN[:, np.newaxis] ** 2 + kE[np.newaxis, :] ** 2)

        amp = np.zeros_like(k_rad)

        if not anisotropic:
            noise_pspec, k, _, _, _, _ = self.powerspecNoise2D()
            k_bin = np.insert(k + k[0] / 2, 0, 0)

            for i in range(k.size):
                k_min = k_bin[i]
                k_max = k_bin[i + 1]
                r = np.logical_and(k_rad > k_min, k_rad <= k_max)
                if i == (k.size - 1):
                    r = k_rad > k_min
                if not np.any(r):
                    continue
                amp[r] = noise_pspec[i]
            amp[k_rad == 0.0] = self.variance
            amp[k_rad > k.max()] = noise_pspec[np.argmax(k)]
            amp = np.sqrt(amp * self.noise_data.size * np.pi * 4)

        elif anisotropic:
            interp_pspec, _, _, _, skE, skN = self.powerspecNoise3D()
            kE = np.fft.fftshift(kE)
            kN = np.fft.fftshift(kN)
            make = np.logical_and(kE >= skE.min(), kE <= skE.max())
            mkN = np.logical_and(kN >= skN.min(), kN <= skN.max())
            mkRad = np.where(  # noqa
                k_rad < np.sqrt(kN[mkN].max() ** 2 + kE[make].max() ** 2)
            )
            res = interp_pspec(kN[mkN, np.newaxis], kE[np.newaxis, make], grid=True)
            amp = res
            amp = np.fft.fftshift(amp)

        spec *= amp
        noise = np.abs(np.fft.ifft2(spec))
        noise -= np.mean(noise)

        # remove shape % 2 padding
        return noise[: shape[0], : shape[1]]

    def getQuadtreeNoise(self, rstate=None, gather=np.nanmedian):
        """Create noise for a :class:`~kite.quadtree.Quadtree`

        Use :meth:`~kite.covariance.Covariance.getSyntheticNoise` to create
        data-driven noise on each quadtree leaf, summarized by

        :param gather: Function gathering leaf's noise realisation,
                       defaults to np.median.
        :type normalisation: callable, optional
        :returns: Array of noise level at each quadtree leaf.
        :rtype: :class:`numpy.ndarray`
        """
        qt = self.quadtree

        syn_noise = self.syntheticNoise(
            shape=self.scene.displacement.shape, rstate=rstate
        )
        syn_noise[self.scene.displacement_mask] = np.nan
        noise_quadtree_arr = np.full(qt.nleaves, np.nan)

        for il, lv in enumerate(qt.leaves):
            noise_quadtree_arr[il] = gather(syn_noise[lv._slice_rows, lv._slice_cols])
        return noise_quadtree_arr

    def powerspecNoise1D(self, data=None, ndeg=512, nk=512):
        if self._powerspec1d_cached is None:
            self._powerspec1d_cached = self._powerspecNoise(
                data, norm="1d", ndeg=ndeg, nk=nk
            )
        return self._powerspec1d_cached

    def powerspecNoise2D(self, data=None, ndeg=512, nk=512):
        if self._powerspec2d_cached is None:
            self._powerspec2d_cached = self._powerspecNoise(
                data, norm="2d", ndeg=ndeg, nk=nk
            )
        return self._powerspec2d_cached

    def powerspecNoise3D(self, data=None):
        if self._powerspec3d_cached is None:
            self._powerspec3d_cached = self._powerspecNoise(data, norm="3d")
        return self._powerspec3d_cached

    def _powerspecNoise(self, data=None, norm="1d", ndeg=512, nk=512):
        """Get the noise power spectrum from
            :attr:`kite.Covariance.noise_data`.

        :param data: Overwrite Covariance.noise_data, defaults to `None`
        :type data: :class:`numpy.ndarray`, optional
        :returns: `(power_spec, k, f_spectrum, kN, kE)`
        :rtype: tuple
        """
        if data is None:
            noise = self.noise_data
        else:
            noise = data.copy()
        if norm not in ("1d", "2d", "3d"):
            raise AttributeError("norm must be 1d, 2d or 3d")

        # noise = squareMatrix(noise)
        shift = np.fft.fftshift

        spectrum = shift(np.fft.fft2(noise, axes=(0, 1), norm=None))
        power_spec = (np.abs(spectrum) / spectrum.size) ** 2

        kE = shift(np.fft.fftfreq(power_spec.shape[1], d=self.quadtree.frame.dEmeter))
        kN = shift(np.fft.fftfreq(power_spec.shape[0], d=self.quadtree.frame.dNmeter))
        k_rad = np.sqrt(kN[:, np.newaxis] ** 2 + kE[np.newaxis, :] ** 2)
        power_spec[k_rad == 0.0] = 0.0

        power_interp = sp.interpolate.RectBivariateSpline(kN, kE, power_spec)

        # def power1d(k):
        #     theta = np.linspace(-np.pi, np.pi, ndeg, False)
        #     power = np.empty_like(k)
        #     for i in range(k.size):
        #         kE = np.cos(theta) * k[i]
        #         kN = np.sin(theta) * k[i]
        #         power[i] = np.median(power_interp.ev(kN, kE)) * k[i]\
        #             * np.pi * 4
        #     return power

        def power1d(k):
            theta = np.linspace(-np.pi, np.pi, ndeg, False)
            power = np.empty_like(k)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            for i in range(k.size):
                kE = cos_theta * k[i]
                kN = sin_theta * k[i]
                power[i] = np.mean(power_interp.ev(kN, kE))

            power *= 2 * np.pi
            return power

        def power2d(k):
            """Mean 2D Power works!"""
            theta = np.linspace(-np.pi, np.pi, ndeg, False)
            power = np.empty_like(k)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            for i in range(k.size):
                kE = sin_theta * k[i]
                kN = cos_theta * k[i]
                power[i] = np.median(power_interp.ev(kN, kE))
                # Median is more stable than the mean here

            return power

        def power3d(k):
            return power_interp

        power = power1d
        if norm == "2d":
            power = power2d
        elif norm == "3d":
            power = power3d

        k_rad = np.sqrt(kN[:, np.newaxis] ** 2 + kE[np.newaxis, :] ** 2)
        k = np.linspace(k_rad[k_rad > 0].min(), k_rad.max(), nk)
        dk = 1.0 / (k[1] - k[0]) / (2 * nk)
        return power(k), k, dk, spectrum, kE, kN

    def _powerCosineTransform(self, p_spec):
        """Calculating the cosine transform of the power spectrum.

        The cosine transform of the power spectrum is an estimate
        of the data covariance (see Hanssen, 2001)."""
        cos = fft.idct(p_spec, type=3)
        return cos

    def setSamplingMethod(self, method):
        """Set the sampling method"""
        assert method in CovarianceConfig.sampling_method.choices

        self.config.sampling_method = method
        self._clear(config=True, spectrum=False)
        self.evChanged.notify()
        self._log.debug("Changed sampling method to %s" % method)

    def setSpatialBins(self, nbins):
        """Set number of spatial bins"""
        self.config.spatial_bins = nbins
        self._clear(config=True, spectrum=False)
        self.evChanged.notify()
        self._log.debug("Changed spatial distance bins to %s" % nbins)

    def setSpatialPairs(self, npairs):
        """Set number of random spatial pairs"""
        self.config.spatial_pairs = npairs
        self._clear(config=True, spectrum=False)
        self.evChanged.notify()
        self._log.debug("Changed random pairs to %s" % npairs)

    def setModelFunction(self, model):
        assert model in CovarianceConfig.model_function.choices
        self.config.model_function = model
        self._clear(config=True, spectrum=True)
        self.evChanged.notify()
        self._log.debug("Changed model function to %s" % model)

    def getModelFunction(self):
        if self.config.model_function == "exponential":
            return modelCovarianceExponential
        if self.config.model_function == "exponential_cosine":
            return modelCovarianceExponentialCosine

    @property_cached
    def covariance_spectral(self):
        """Covariance function estimated directly from the power spectrum of
            displacement noise patch using the cosine transform.

        :type: tuple, :class:`numpy.ndarray` (covariance, distance)"""
        power_spec, k, dk, _, _, _ = self.powerspecNoise1D()
        # power_spec -= self.variance

        d = np.arange(1, power_spec.size + 1) * dk
        cov = self._powerCosineTransform(power_spec)

        return cov, d

    @property_cached
    def covariance_spatial(self):
        self._log.debug("Estimating covariance (spatial)...")

        nbins = self.config.spatial_bins
        npairs = self.config.spatial_pairs
        noise_data = self.noise_data.ravel()
        noise_data -= noise_data.mean()

        grdE = self.noise_data_gridE
        grdN = self.noise_data_gridN

        max_distance = min(abs(grdE.min() - grdE.max()), abs(grdN.min() - grdN.max()))
        dist_bins = np.linspace(0, max_distance, nbins + 1)

        grdE = grdE.ravel()
        grdN = grdN.ravel()

        # Select random coordinates
        rstate = np.random.RandomState(noise_data.size)
        rand_idx = rstate.randint(0, noise_data.size, (2, npairs))
        idx0 = rand_idx[0, :]
        idx1 = rand_idx[1, :]

        distances = np.sqrt(
            (grdN[idx0] - grdN[idx1]) ** 2 + (grdE[idx0] - grdE[idx1]) ** 2
        )

        cov_all = noise_data[idx0] * noise_data[idx1]
        vario_all = (noise_data[idx0] - noise_data[idx1]) ** 2

        bins = np.digitize(distances, dist_bins, right=True)
        bin_distances = dist_bins[1:] - dist_bins[1] / 2

        covariance = np.full_like(bin_distances, fill_value=np.nan)
        variance = np.full_like(bin_distances, fill_value=np.nan)

        for ib in range(nbins):
            selection = bins == ib
            if selection.sum() != 0:
                covariance[ib] = np.nanmean(cov_all[selection])
                variance[ib] = np.nanmean(vario_all[selection]) / 2

        self._structure_spatial = (
            variance[~np.isnan(variance)],
            bin_distances[~np.isnan(variance)],
        )
        covariance[0] = np.nan
        return (
            covariance[~np.isnan(covariance)],
            bin_distances[~np.isnan(covariance)],
        )

    def getCovariance(self):
        """Calculate the covariance function

        :return: The covariance and distance
        :rtype: tuple
        """
        if self.config.sampling_method == "spatial":
            return self.covariance_spatial
        elif self.config.sampling_method == "spectral":
            return self.covariance_spectral

    @property
    def covariance_model(self, regime=0):
        """Covariance model parameters for
            :func:`~kite.covariance.modelCovariance` retrieved
            from :attr:`~kite.Covariance.getCovarianceFunction`.

        .. note:: using this function implies several model
            fits: (1) fit of the spectrum and (2) the cosine transform.
            Not sure about the consequences, if this is useful and/or
            meaningful.

        :getter: Get the parameters.
        :type: tuple, ``a`` and ``b``
        """
        if self.config.model_coefficients is None:
            covariance, distance = self.getCovariance()
            model = self.getModelFunction()

            if self.config.model_function == "exponential":
                coeff = (np.mean(covariance), np.mean(distance))

            elif self.config.model_function == "exponential_cosine":
                coeff = (
                    np.mean(covariance),
                    np.mean(distance),
                    np.mean(distance) * -0.1,
                    0.1,
                )

                func = self.getModelFunction()

                # Testing penalty function
                def model(*args):
                    distance, a, b, c, d = args
                    res = func(*args)

                    penalty = 0.0
                    if distance[-1] / b > (distance[-1] + c) / d:
                        penalty = (b - d) * coeff[0]
                        self._log.warning("Penalty %f" % penalty)

                    return res + penalty

                # Overwrite with pure model function
                model = self.getModelFunction()  # noqa

            try:
                coeff, _ = sp.optimize.curve_fit(model, distance, covariance, p0=coeff)
            except (RuntimeError, TypeError) as e:
                self._log.exception(e)
                self._log.warning(
                    "Could not fit the %s covariance model", self.config.model_function
                )
            finally:
                self.config.model_coefficients = tuple(map(float, coeff))

        return self.config.model_coefficients

    @property
    def covariance_model_rms(self):
        """
        :getter: RMS missfit between :class:`~kite.Covariance.covariance_model`
            and :class:`~kite.Covariance.getCovarianceFunction`
        :type: float
        """
        cov, d = self.getCovariance()
        model = self.getModelFunction()
        cov_mod = model(d, *self.covariance_model)

        return np.sqrt(np.mean((cov - cov_mod) ** 2))

    @property_cached
    def structure_spatial(self):
        self.covariance_spatial
        return self._structure_spatial

    @property_cached
    def structure_spectral(self):
        """Structure function derived from ``noise_patch``
            :type: tuple, :class:`numpy.ndarray` (structure_spectral, distance)

        Adapted from
        http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
        """
        power_spec, k, dk, _, _, _ = self.powerspecNoise1D()
        d = np.arange(1, power_spec.size + 1) * dk

        def structure_spectral(power_spec, d, k):
            struc_func = np.zeros_like(k)
            for i, d in enumerate(d):
                for ik, tk in enumerate(k):
                    # struc_func[i] += (1. - np.cos(tk*d))*power_spec[ik]
                    struc_func[i] += (1.0 - sp.special.j0(tk * d)) * power_spec[ik]
            struc_func *= 2.0 / 1
            return struc_func

        struc_func = structure_spectral(power_spec, d, k)
        return struc_func, d

    def getStructure(self, method=None):
        """Get the structure function

        :param method: Either `spatial` or `spectral`, if `None`
            the method is taken from config
        :type method: str (optional)

        :return: (variance, distance)
        :rtype: tuple
        """
        if method is None:
            method = self.config.sampling_method
        if method == "spatial":
            return self.structure_spatial
        elif method == "spectral":
            return self.structure_spectral

    @property
    def variance(self):
        """Variance of data noise estimated from the
            high-frequency end of power spectrum.

        :setter: Set the variance manually
        :getter: Retrieve the variance
        :type: float
        """
        return self.config.variance

    @variance.setter
    def variance(self, value):
        self.config.variance = float(value)
        # self._clear(config=False, spectrum=False, spatial=False)
        self.evChanged.notify()

    @variance.getter
    def variance(self):
        if self.config.variance is None and self.config.sampling_method == "spatial":
            structure_spatial, dist = self.structure_spatial

            last_20p = -int(structure_spatial.size * 0.2)
            self.config.variance = float(np.mean(structure_spatial[(last_20p):]))

        elif self.config.variance is None and self.config.sampling_method == "spectral":
            power_spec, k, dk, spectrum, _, _ = self.powerspecNoise1D()
            cov, _ = self.covariance_spectral
            ma = self.covariance_model[0]
            # print(cov[1])
            ps = power_spec * spectrum.size
            # print(spectrum.size)
            # print(np.mean(ps[-int(ps.size/9.):-1]))
            var = np.median(ps[-int(ps.size / 9.0) :]) + ma
            self.config.variance = float(var)

        return self.config.variance

    def export_weight_matrix(self, filename):
        """Export the full :attr:`~kite.Covariance.weight_matrix` to an ASCII
            file. The data can be loaded through :func:`numpy.loadtxt`.

        :param filename: path to export to
        :type filename: str
        """
        self._log.debug("Exporting Covariance.weight_matrix to %s" % filename)
        header = (
            "Exported kite.Covariance.weight_matrix, "
            "for more information visit https://pyrocko.org\n"
            "\nThe matrix is symmetric and ordered by QuadNode.id:\n"
        )
        header += ", ".join([lv.id for lv in self.quadtree.leaves])
        np.savetxt(filename, self.weight_matrix, header=header)

    def get_state_hash(self):
        sha = sha1()
        sha.update(str(self.config).encode())
        return sha.digest().hex()

    @property_cached
    def plot(self):
        """Simple overview plot to summarize the covariance estimations."""
        from kite.plot2d import CovariancePlot

        return CovariancePlot(self)

    @property_cached
    def plot_syntheticNoise(self):
        """Simple overview plot to summarize the covariance estimations."""
        from kite.plot2d import SyntheticNoisePlot

        return SyntheticNoisePlot(self)
