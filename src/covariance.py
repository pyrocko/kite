#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as num
import scipy as sp
import time

import covariance_ext
from pyrocko import guts
from pyrocko.guts_array import Array
from kite.meta import (Subject, property_cached,  # noqa
                       trimMatrix, derampMatrix, squareMatrix)

__all__ = ['Covariance', 'CovarianceConfig']

noise_regimes = [
    (1./2000, num.inf),
    (1./2000, 1./500),
    (1./500, 1./10),
    (0, num.inf)]


def modelCovariance(distance, a, b):
    """Exponential model to estimate a positive-definite covariance

    We assume the following covariance model to describe the empirical noise
    observation:

    .. math::

        cov(dist) = c \\cdot e^{\\frac{-dist}{b}}

    :param distance: Distance between
    :type distance: float or :class:`numpy.ndarray`
    :param a: Linear model parameter
    :type a: float
    :param b: Exponential model parameter
    :type b: float
    :returns: Covariance at ``distance``
    :rtype: :class:`numpy.ndarray`
    """
    return a * num.exp(-distance/b)


def modelPowerspec(k, beta, D):
    """Exponential Linear model to estimate a log-linear powerspectrum

    We assume the following log-linear model for the measured powerspectrum

    .. math::

        pow(k) = \\frac{k^\\beta}{D}


    :param k: Wavenumber
    :type k: float or :class:`numpy.ndarray`
    :param a: Exponential model factor
    :type a: float
    :param b: Fractional model factor
    :type b: float
    """
    return (k**beta)/D


class CovarianceConfig(guts.Object):
    noise_coord = Array.T(
        shape=(None,), dtype=num.float,
        serialize_as='list',
        optional=True,
        help='Noise patch coordinates and size,')
    a = guts.Float.T(
        optional=True,
        help='Exponential covariance model; scaling factor. '
             'See :func:`~kite.covariance.modelCovariance`')
    b = guts.Float.T(
        optional=True,
        help='Exponential covariance model; exponential decay. '
             'See :func:`~kite.covariance.modelCovariance`')
    variance = guts.Float.T(
        optional=True,
        help='Variance of the model')
    adaptive_subsampling = guts.Bool.T(
        default=True,
        help='Adaptive subsampling flag for full covariance calculation.')
    covariance_matrix = Array.T(
        optional=True,
        serialize_as='base64',
        help='Cached covariance matrix, '
             'see :attr:`~kite.Covariance.covariance_matrix`',
        )


class Covariance(object):
    """Analytical covariance for noise estimation of
    :class:`kite.Scene.displacement`.

    The covariance between :attr:`kite.Quadtree.leafs` is
    used as a weighting measure for the optimization process.

    Two different methods are implemented to estimate the covariance function:

    1. The distance between :class:`~kite.quadtree.QuadNode`
       leaf focal points (:attr:`~kite.Covariance.covariance_focal`)
    2. The more *accurate* statistical distances between every nodes pixels,
       this process is computational very expensive and
       can take a few minutes or longer.
       See :class:`~kite.Covariance.covariance` or
       :class:`~kite.Covariance.weight_matrix`.

    :param quadtree: Quadtree to work on
    :type quadtree: :class:`~kite.Quadtree`
    :param config: Config object
    :type config: :class:`~kite.covariance.CovarianceConfig`
    """
    evChanged = Subject()
    evConfigChanged = Subject()

    def __init__(self, scene, config=CovarianceConfig()):
        self.frame = scene.frame
        self.quadtree = scene.quadtree
        self.scene = scene
        self._noise_data = None
        self._powerspec1d_cached = None
        self._powerspec2d_cached = None
        self._powerspec3d_cached = None
        self._initialized = False
        self._nthreads = 0
        self._log = scene._log.getChild('Covariance')

        self.setConfig(config)
        self.quadtree.evChanged.subscribe(self._clear)
        self.scene.evConfigChanged.subscribe(self.setConfig)

    def __call__(self, *args, **kwargs):
        return self.getLeafCovariance(*args, **kwargs)

    def setConfig(self, config=None):
        """ Sets and updated the config of the instance

        :param config: New config instance, defaults to configuration provided
                       by parent :class:`~kite.Scene`
        :type config: :class:`~kite.covariance.CovarianceConfig`, optional
        """
        if config is None:
            config = self.scene.config.covariance

        self.config = config

        if config.noise_coord is None\
           and (config.a is not None or
                config.b is not None or
                config.variance is not None):
            self.noise_data  # init data array
            self.config.a = config.a
            self.config.b = config.b
            self.config.variance = config.variance

        self._clear(config=False)
        self.evConfigChanged.notify()

    def _clear(self, config=True, spectrum=True):
        if config:
            self.config.a = None
            self.config.b = None
            self.config.variance = None
            self.config.covariance_matrix = None

        if spectrum:
            self.structure_func = None
            self._powerspec1d_cached = None
            self._powerspec2d_cached = None

        self.covariance_matrix = None
        self.covariance_matrix_focal = None
        self.covariance_func = None
        self.weight_matrix = None
        self.weight_matrix_focal = None
        self._initialized = False
        self.evChanged.notify()

    @property
    def nthreads(self):
        ''' Number of threads (CPU cores) to use for full covariance
            calculation

        Setting ``nthreads`` to ``0`` uses all available cores (default).

        :setter: Sets the number of threads
        :type: int
        '''
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = int(value)

    @property
    def noise_coord(self):
        """ Coordinates of the noise patch in local coordinates.

        :setter: Set the noise coordinates
        :getter: Get the noise coordinates
        :type: :class:`numpy.ndarray`, ``[llE, llN, sizeE, sizeN]``
        """
        if self.config.noise_coord is None:
            self.noise_data
        return self.config.noise_coord

    @noise_coord.setter
    def noise_coord(self, values):
        self.config.noise_coord = num.array(values)

    @property
    def noise_patch_size_km2(self):
        '''
        :getter: Noise patch size in :math:`km^2`.
        :type: float
        '''
        if self.noise_coord is None:
            return 0.
        size = (self.noise_coord[2] * self.noise_coord[3])*1e-6
        if size < 75:
            self._log.warning('Defined noise patch is instably small')
        return size

    @property
    def noise_data(self, data):
        ''' Noise data we process to estimate the covariance

        :setter: Set the noise patch to analyze the covariance.
        :getter: If the noise data has not been set manually, we grab data
                 through :func:`~kite.Covariance.selectNoiseNode`.
        :type: :class:`numpy.ndarray`
        '''
        return self._noise_data

    @noise_data.getter
    def noise_data(self):
        if self._noise_data is not None:
            return self._noise_data
        elif self.config.noise_coord is not None:
            self._log.info('Selecting noise_data from config...')
            llE, llN = self.scene.frame.mapENMatrix(
                *self.config.noise_coord[:2])
            sE, sN = self.scene.frame.mapENMatrix(
                *self.config.noise_coord[2:])
            slice_E = slice(llE, llE + sE)
            slice_N = slice(llN, llN + sN)
            self.noise_data = self.scene.displacement[slice_N, slice_E]
        else:
            self._log.info('Selecting noise_data from Quadtree...')
            node = self.selectNoiseNode()
            self.noise_data = node.displacement
            self.noise_coord = [node.llE, node.llN,
                                node.sizeE, node.sizeN]
        return self.noise_data

    @noise_data.setter
    def noise_data(self, data):
        data = data.copy()
        data = derampMatrix(trimMatrix(data))
        data = trimMatrix(data)
        data[num.isnan(data)] = 0.
        self._noise_data = data
        self._clear()

    def selectNoiseNode(self):
        """ Choose noise node from quadtree
        the biggest :class:`~kite.quadtree.QuadNode` from
        :class:`~kite.Quadtree`.

        :returns: A quadnode with the least signal.
        :rtype: :class:`~kite.quadtree.QuadNode`
        """
        t0 = time.time()

        stdmax = max([n.std for n in self.quadtree.nodes])  # noqa
        lmax = max([n.std for n in self.quadtree.nodes])  # noqa

        def costFunction(n):
            nl = num.log2(n.length)/num.log2(lmax)
            ns = n.std/stdmax
            return nl*(1.-ns)*(1.-n.nan_fraction)

        nodes = sorted(self.quadtree.nodes,
                       key=costFunction)

        self._log.debug('Fetched noise from Quadtree.nodes [%0.8f s]'
                        % (time.time() - t0))
        return nodes[0]

    def _mapLeafs(self, nx, ny):
        """ Helper function returning appropriate
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
        leaf1 = self.quadtree.leafs[nx]
        leaf2 = self.quadtree.leafs[ny]

        self._leaf_mapping[leaf1.id] = nx
        self._leaf_mapping[leaf2.id] = ny

        return leaf1, leaf2

    @property_cached
    def covariance_matrix(self):
        """ Covariance matrix calculated from sub-distances pairs from quadtree
            node-to-node.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs` x
            :class:`~kite.Quadtree.nleafs`)
        """
        if not isinstance(self.config.covariance_matrix, num.ndarray):
            self.config.covariance_matrix =\
                self._calcCovarianceMatrix(method='full')
        elif self.config.covariance_matrix.ndim == 1:
            try:
                nl = self.quadtree.nleafs
                self.config.covariance_matrix =\
                    self.config.covariance_matrix.reshape(nl, nl)
            except ValueError:
                self.config.covariance = None
                return self.covariance_matrix
        return self.config.covariance_matrix

    @property_cached
    def covariance_matrix_focal(self):
        """ This matrix uses distances between focal points. Fast but
            statistically not reliable method. For final approach use
            :attr:`~kite.Covariance.covariance_matrix`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs` x
            :class:`~kite.Quadtree.nleafs`)
        """
        return self._calcCovarianceMatrix(method='focal')

    @property_cached
    def weight_matrix(self):
        """ Weight matrix from full covariance :math:`\\sqrt{cov^{-1}}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs` x
            :class:`~kite.Quadtree.nleafs`)
        """
        return num.linalg.inv(self.covariance_matrix)

    @property_cached
    def weight_matrix_focal(self):
        """ Weight matrix from fast focal method
            :math:`\\sqrt{cov_{focal}^{-1}}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs` x
            :class:`~kite.Quadtree.nleafs`)
        """
        return num.linalg.inv(self.covariance_matrix_focal)

    @property_cached
    def weight_vector(self):
        """ Weight vector from full covariance :math:`\\sqrt{cov^{-1}}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs`)
        """
        return num.sum(self.weight_matrix, axis=1)

    @property_cached
    def weight_vector_focal(self):
        """ Weight vector from fast focal method
            :math:`\\sqrt{cov_{focal}^{-1}}`.
        :type: :class:`numpy.ndarray`,
            size (:class:`~kite.Quadtree.nleafs`)
        """
        return num.sum(self.weight_matrix_focal, axis=1)

    def _calcCovarianceMatrix(self, method='focal'):
        """Calculates the covariance matrix.

        :param method: Either ``focal`` point distances are used - this is
            quick but statistically not reliable.
            Or ``full``, where the full quadtree pixel distances matrices are
            calculated
            , defaults to ``focal``
        :type method: str, optional
        :returns: Covariance matrix
        :rtype: thon:numpy.ndarray
        """
        self._initialized = True

        nl = len(self.quadtree.leafs)
        self._leaf_mapping = {}

        t0 = time.time()
        ma, mb = self.covariance_model
        if method == 'focal':
            dist_matrix = num.zeros((nl, nl))
            dist_iter = num.nditer(num.triu_indices_from(dist_matrix))

            for nx, ny in dist_iter:
                leaf1, leaf2 = self._mapLeafs(nx, ny)
                dist = self._leafFocalDistance(leaf1, leaf2)
                dist_matrix[(nx, ny), (ny, nx)] = dist
            cov_matrix = modelCovariance(dist_matrix, ma, mb)

        elif method == 'full':
            leaf_map = num.empty((len(self.quadtree.leafs), 4),
                                 dtype=num.uint32)
            for nl, leaf in enumerate(self.quadtree.leafs):
                leaf, _ = self._mapLeafs(nl, nl)
                leaf_map[nl, 0], leaf_map[nl, 1] = (leaf._slice_rows.start,
                                                    leaf._slice_rows.stop)
                leaf_map[nl, 2], leaf_map[nl, 3] = (leaf._slice_cols.start,
                                                    leaf._slice_cols.stop)

            nleafs = self.quadtree.nleafs
            cov_matrix = covariance_ext.covariance_matrix(
                            self.scene.frame.gridE.filled(),
                            self.scene.frame.gridN.filled(),
                            leaf_map, ma, mb, self.nthreads,
                            self.config.adaptive_subsampling)\
                .reshape(nleafs, nleafs)
        else:
            raise TypeError('Covariance calculation %s method not defined!'
                            % method)

        num.fill_diagonal(cov_matrix, self.variance)
        self._log.debug('Created covariance matrix - %s mode [%0.8f s]' %
                        (method, time.time()-t0))
        return cov_matrix

    @staticmethod
    def _leafFocalDistance(leaf1, leaf2):
        return num.sqrt((leaf1.focal_point[0]
                         - leaf2.focal_point[0])**2 +
                        (leaf1.focal_point[1]
                         - leaf2.focal_point[1])**2)

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
            raise KeyError('Unknown quadtree leaf with id %s' % e)

    def getLeafCovariance(self, leaf1, leaf2):
        """Get the covariance between ``leaf1`` and ``leaf2``.

        :param leaf1: Leaf one
        :type leaf1: str of `leaf.id` or :class:`~kite.quadtree.QuadNode`
        :param leaf2: Leaf two
        :type leaf2: str of `leaf.id` or :class:`~kite.quadtree.QuadNode`
        :returns: Covariance between ``leaf1`` and ``leaf2``
        :rtype: float
        """
        return self.covariance_matrix[self._leafMapping(leaf1, leaf2)]

    def getLeafWeight(self, leaf, model='focal'):
        ''' Get the absolute weight of ``leaf``, summation of all weights from
        :attr:`kite.Covariance.weight_matrix`

        .. math ::

            w_{x} = \\sum_i W_{x,i}

        :param model: ``Focal`` or ``full``, default ``focal``
        :type model: str
        :param leaf: A leaf from :class:`~kite.Quadtree`
        :type leaf: :class:`~kite.quadtree.QuadNode`

        :returns: Weight of the leaf
        :rtype: float
        '''
        (nl, _) = self._leafMapping(leaf, leaf)
        weight_mat = self.weight_matrix_focal
        return num.mean(weight_mat, axis=0)[nl]

    def syntheticNoise(self, shape=(1024, 1024), dEdN=None,
                       anisotropic=False):
        """Create random synthetic noise with the same character as defined
            in :attr:`noise_data`.

        This function uses the powerspectrum of the empirical noise
        (:func:`powerspecNoise`) to create synthetic noise for model
        pertubation. The default sampling distances are taken from
        :attr:`kite.scene.Frame.dE` and :attr:`kite.scene.Frame.dN`. And can be
        overwritten.
        :param shape: shape of the desired noise patch.
            Pixels in northing and easting (`nE`, `nN`),
            defaults to `(1024, 1024)`.
        :type shape: tuple, optional
        :param dEdN: The sampling distance in easting, defaults to
            (:attr:`kite.scene.Frame.dE`, :attr:`kite.scene.Frame.dN`).
        :type dE: tuple, floats
        :returns: Noise patch
        :rtype: :class:`numpy.ndarray`
        """
        if (shape[0] + shape[1]) % 2 != 0:
            # self._log.warning('Patch dimensions must be even, '
            #                   'ceiling dimensions!')
            pass
        nE = shape[1] + (shape[1] % 2)
        nN = shape[0] + (shape[0] % 2)

        rfield = num.random.rand(nN, nE)
        spec = num.fft.fft2(rfield)

        if not dEdN:
            dE, dN = (self.scene.frame.dE, self.scene.frame.dN)
        kE = num.fft.fftfreq(nE, dE)
        kN = num.fft.fftfreq(nN, dN)
        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

        amp = num.zeros_like(k_rad)

        if not anisotropic:
            noise_pspec, k, _, _, _, _ = self.powerspecNoise1D()
            k_bin = num.insert(k + k[0]/2, 0, 0)

            for i in xrange(k.size):
                k_min = k_bin[i]
                k_max = k_bin[i+1]
                r = num.logical_and(k_rad > k_min, k_rad <= k_max)
                if i == (k.size-1):
                    r = k_rad > k_min
                if r.sum() == 0:
                    continue
                amp[r] = noise_pspec[i]
            amp[k_rad == 0.] = self.variance
            amp[k_rad > k.max()] = noise_pspec[num.argmax(k)]
            amp = num.sqrt(amp)
            # amp = num.sqrt(amp * max(self.noise_data.shape)**2)

        elif anisotropic:
            interp_pspec, _, _, _, skE, skN = self.powerspecNoise3D()
            kE = num.fft.fftshift(kE)
            kN = num.fft.fftshift(kN)
            mkE = num.logical_and(kE >= skE.min(), kE <= skE.max())
            mkN = num.logical_and(kN >= skN.min(), kN <= skN.max())
            mkRad = num.where(  # noqa
                k_rad < num.sqrt(kN[mkN].max()**2 + kE[mkE].max()**2))
            res = interp_pspec(kN[mkN, num.newaxis],
                               kE[num.newaxis, mkE], grid=True)
            print amp.shape, res.shape
            print kN.size, kE.size
            amp = res
            amp = num.fft.fftshift(amp)
            print amp.min(), amp.max()

        spec *= amp
        noise = num.abs(num.fft.ifft2(spec))
        noise -= num.mean(noise)
        return noise

    def powerspecNoise1D(self, data=None, ndeg=512, nk=512):
        if self._powerspec1d_cached is None:
            self._powerspec1d_cached = self._powerspecNoise(
                data, norm='1d', ndeg=ndeg, nk=nk)
        return self._powerspec1d_cached

    def powerspecNoise2D(self, data=None, ndeg=512, nk=512):
        if self._powerspec2d_cached is None:
            self._powerspec2d_cached = self._powerspecNoise(
                data, norm='2d', ndeg=ndeg, nk=nk)
        return self._powerspec2d_cached

    def powerspecNoise3D(self, data=None):
        if self._powerspec3d_cached is None:
            self._powerspec3d_cached = self._powerspecNoise(
                data, norm='3d')
        return self._powerspec3d_cached

    def _powerspecNoise(self, data=None, norm='1d', ndeg=512, nk=512):
        """Get the noise spectrum from
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
        if norm not in ['1d', '2d', '3d']:
            raise AttributeError('norm must be either 1d, 2d or 3d')

        # noise = squareMatrix(noise)
        shift = num.fft.fftshift

        spectrum = shift(num.fft.fft2(noise, axes=(0, 1), norm=None))
        power_spec = (num.abs(spectrum)/spectrum.size)**2

        kE = shift(num.fft.fftfreq(power_spec.shape[1],
                                   d=self.quadtree.frame.dE))
        kN = shift(num.fft.fftfreq(power_spec.shape[0],
                                   d=self.quadtree.frame.dN))
        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)
        power_spec[k_rad == 0.] = 0.

        power_interp = sp.interpolate.RectBivariateSpline(kN, kE, power_spec)

        # def power1d(k):
        #     theta = num.linspace(-num.pi, num.pi, ndeg, False)
        #     power = num.empty_like(k)
        #     for i in xrange(k.size):
        #         kE = num.cos(theta) * k[i]
        #         kN = num.sin(theta) * k[i]
        #         power[i] = num.median(power_interp.ev(kN, kE)) * k[i]\
        #             * num.pi * 4
        #     return power

        def power1d(k):
            theta = num.linspace(-num.pi, num.pi, ndeg, False)
            power = num.empty_like(k)
            for i in xrange(k.size):
                kE = num.cos(theta) * k[i]
                kN = num.sin(theta) * k[i]
                power[i] = num.median(power_interp.ev(kN, kE))
            return (power * num.pi * 4)

        def power2d(k):
            """ Mean 2D Power works! """
            theta = num.linspace(-num.pi, num.pi, ndeg, False)
            power = num.empty_like(k)
            for i in xrange(k.size):
                kE = num.sin(theta) * k[i]
                kN = num.cos(theta) * k[i]
                power[i] = num.mean(power_interp.ev(kN, kE))
                # Median is more stable than the mean here
            return power

        def power3d(k):
            return power_interp

        power = power1d
        if norm == '2d':
            power = power2d
        elif norm == '3d':
            power = power3d

        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)
        k = num.linspace(k_rad[k_rad > 0].min(),
                         k_rad.max(), nk)
        dk = 1./k.min() / (2. * nk)
        return power(k), k, dk, spectrum, kE, kN

        def power1Ddisc():
            self._log.info('Using discrete summation')
            ps = power_spec
            d = num.abs(num.arange(-ps.shape[0]/2,
                                   ps.shape[0]/2))
            rm = num.sqrt(d[:, num.newaxis]**2 + d[num.newaxis, :]**2)

            axis = num.argmax(ps.shape)
            k_ref = kN if axis == 0 else kE
            p = num.empty(ps.shape[axis]/2)
            k = num.empty(ps.shape[axis]/2)
            for r in xrange(ps.shape[axis]/2):
                mask = num.logical_and(rm >= r-.5, rm < r+.5)
                k[r] = k_ref[(k_ref.size/2)+r]
                p[r] = num.median(ps[mask]) * 4 * num.pi
            return p, k

        power, k = power1Ddisc()
        dk = k[1] - k[0]
        return power, k, dk, spectrum, kE, kN

    def _powerspecFit(self, regime=3):
        power_spec, k, _, _, _, _ = self.powerspecNoise1D()

        def selectRegime(k, k1, k2):
            return num.logical_and(k > k1, k < k2)

        regime = selectRegime(k, *noise_regimes[regime])

        try:
            return sp.optimize.curve_fit(modelPowerspec,
                                         k[regime], power_spec[regime],
                                         p0=(self.variance, 2000))
        except RuntimeError:
            self._log.warning('Could not fit the powerspectrum model.')
            return (0., 0.), 0.

    @property
    def powerspec_model(self):
        """Powerspectrum model parameters based on the spectral model after
        :func:`~kite.covariance.modelPowerspec`

        :returns: Model parameters ``a`` and ``b``
        :rtype: tuple, floats
        """
        p, _ = self._powerspecFit()
        return p

    @property
    def powerspec_model_rms(self):
        '''
        :getter: RMS missfit between :class:`~kite.Covariance.powerspecNoise1D`
            and :class:`~kite.Covariance.powerspec_model``
        :type: float
        '''
        power_spec, k, _, _, _, _ = self.powerspecNoise1D()
        power_spec_mod = self.powerspecModel(k)
        return num.sqrt(num.mean((power_spec - power_spec_mod)**2))

    def powerspecModel(self, k):
        ''' Calculates the analytical power based on the fit of
        :func:`~kite.covariance.powerspec_model`.

        :param k: Wavenumber(s)
        :type k: float or :class:`numpy.ndarray`
        :returns: Power at wavenumber ``k``
        :rtype: float or :class:`numpy.ndarray`
        '''
        p = self.powerspec_model
        return modelPowerspec(k, *p)

    def _powerCosineTransform(self, p_spec):
        cos = sp.fftpack.idct(p_spec, type=3)
        return cos

    @property_cached
    def covariance_func(self):
        ''' Covariance function derived from powerspectrum of
            displacement noise patch.
        :type: tuple, :class:`numpy.ndarray` (covariance, distance) '''
        power_spec, k, dk, _, _, _ = self.powerspecNoise1D()
        # power_spec -= self.variance

        d = num.arange(1, power_spec.size+1) * dk
        cov = self._powerCosineTransform(power_spec)

        return cov, d

    def covarianceAnalytical(self, regime=0):
        ''' Analytical covariance based on the spectral model fit
        from :func:`~kite.covariance.modelPowerspec`

        :return: Covariance and corresponding distances.
        :rtype: tuple, :class:`numpy.ndarray` (covariance_analytical, distance)
        '''
        _, k, dk, _, kN, kE = self.powerspecNoise1D()
        (a, b) = self.powerspec_model

        spec = modelPowerspec(k, a, b)
        d = num.arange(1, spec.size+1) * dk

        cos = self._powerCosineTransform(spec)
        return cos, d

    @property
    def covariance_model(self, regime=0):
        ''' Covariance model parameters for
        :func:`~kite.covariance.modelCovariance` retrieved
        from :attr:`~kite.Covariance.covarianceAnalytical`.

        :getter: Get the parameters.
        :type: tuple, ``a`` and ``b``
        '''
        if self.config.a is None or self.config.b is None:
            cov, d = self.covarianceAnalytical(regime)
            cov, d = self.covariance_func
            try:
                (a, b), _ =\
                    sp.optimize.curve_fit(modelCovariance, d, cov,
                                          p0=(.001, 500.))
                self.config.a, self.config.b = (float(a), float(b))
            except RuntimeError:
                self._log.warning('Could not fit the covariance model')
                self.config.a, self.config.b = (1., 1000.)
        return self.config.a, self.config.b

    @property
    def covariance_model_rms(self):
        '''
        :getter: RMS missfit between :class:`~kite.Covariance.covariance_model`
            and :class:`~kite.Covariance.covariance_func`
        :type: float
        '''
        cov, d = self.covariance_func
        cov_mod = modelCovariance(d, *self.covariance_model)
        return num.sqrt(num.mean((cov - cov_mod)**2))

    @property_cached
    def structure_func(self):
        ''' Structure function derived from ``noise_patch``
        :type: tuple, :class:`numpy.ndarray` (structure_func, distance)

        Adapted from
        http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
        '''
        power_spec, k, dk, _, _, _ = self.powerspecNoise1D()
        d = num.arange(1, power_spec.size+1) * dk

        def structure_func(power_spec, d, k):
            struc_func = num.zeros_like(k)
            for i, d in enumerate(d):
                for ik, tk in enumerate(k):
                    # struc_func[i] += (1. - num.cos(tk*d))*power_spec[ik]
                    struc_func[i] += (1. - sp.special.j0(tk*d))*power_spec[ik]
            struc_func *= 2./1
            return struc_func

        struc_func = structure_func(power_spec, d, k)
        return struc_func, d

    @property
    def variance(self):
        ''' Variance, derived from the mean of
        :attr:`~kite.Covariance.structure_func`.

        :setter: Set the variance manually
        :getter: Retrieve the variance
        :type: float
        '''
        return self.config.variance

    @variance.setter
    def variance(self, value):
        self.config.variance = float(value)
        self._clear(config=False, spectrum=False)
        self.evChanged.notify()

    @variance.getter
    def variance(self):
        if self.config.variance is None:
            power_spec, k, dk, spectrum, _, _ = self.powerspecNoise1D()
            cov, _ = self.covariance_func
            # print cov[1]
            ps = power_spec * spectrum.size
            # print spectrum.size
            # print num.mean(ps[-int(ps.size/9.):-1])
            var = num.mean(ps[-int(ps.size/9.):-1]) + cov[1]
            self.config.variance = float(var)
        return self.config.variance

    def export_weight_matrix(self, filename):
        """ Export the full :attr:`~kite.Covariance.weight_matrix` to an ASCII
        file. The data can be loaded through :func:`numpy.loadtxt`.

        :param filename: path to export to
        :type filename: str
        """
        self._log.debug('Exporting Covariance.weight_matrix to %s' % filename)
        header = 'Exported kite.Covariance.weight_matrix, '\
                 'for more information visit http://pyrocko.com\n'\
                 '\nThe matrix is symmetric and ordered by QuadNode.id:\n'
        header += ', '.join([l.id for l in self.quadtree.leafs])
        num.savetxt(filename, self.weight_matrix, header=header)

    @property_cached
    def plot(self):
        ''' Simple overview plot to summarize the covariance. '''
        from kite.plot2d import CovariancePlot
        return CovariancePlot(self)
