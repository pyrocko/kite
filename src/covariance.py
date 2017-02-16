#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as num
import scipy as sp
import time

import covariance_ext
from pyrocko import guts
from pyrocko.guts_array import Array
from kite.meta import Subject, property_cached, trimMatrix, derampMatrix

__all__ = ['Covariance', 'CovarianceConfig']

noise_regimes = [
    (1./2000, num.inf),
    (1./2000, 1./500),
    (1./500, 1./10),
    (0, num.inf)]


def modelCovariance(distance, a, b):
    """Exponential function model to approximate a positive-definite covariance

    We assume the following simple covariance model to describe the empirical noise
    observations:

    .. math::

        cov(dist) = a \\cdot e^{\\frac{-dist}{b}}

    :param distance: Distance between
    :type distance: float or :py:class:`numpy.ndarray`
    :param a: Linear model parameter
    :type a: float
    :param b: Exponential model parameter
    :type b: float
    :returns: Covariance at ``distance``
    :rtype: :py:class:`numpy.ndarray`
    """
    return a * num.exp(-distance/b)


def modelPowerspec(k, a, b):
    """Exponential linear model to estimate a log-linear power spectrum

    We assume the following log-linear model for a measured power spectrum

    .. math::

        pow(k) = k^a * \\frac{1}{b}


    :param k: Wavenumber
    :type k: float or :py:class:`numpy.ndarray`
    :param a: Exponential model factor
    :type a: float
    :param b: Fractional model factor
    :type b: float
    """
    return (k**a)/b


class CovarianceConfig(guts.Object):
    noise_coord = Array.T(shape=(None,), dtype=num.float,
                          serialize_as='list',
                          default=[num.nan, num.nan, num.nan, num.nan])
    a = guts.Float.T(default=-9999.,
                     help='Exponential covariance model - scaling factor')
    b = guts.Float.T(default=-9999.,
                     help='Exponential covariance model - exponential decay')
    variance = guts.Float.T(default=-9999., help='Variance')


class Covariance(object):
    """Construct the variance-covariance matrix of quadtree subsampled data.

    Variance and covariance estimates are used to construct the weighting
    matrix to be used later in an optimization. 
    
    Two different methods exist to propagate full-resolution data variances
    and covariances to the covariance matrix of the subsampled dataset:
    
    the covariance function
    The distance between either the :py:class:`kite.quadtree.Quadtree` 
    leafs  
    used as a weighting measure for the optimization process.

    Two different methods exist to estimate the covariance function

    1. The distance between :py:class:`kite.quadtree.QuadNode`
       leaf focal points, :py:class:`kite.covariance.Covariance.matrix_focal`
       defines the approximate covariance of the quadtree leaf pair.
    2. The _accurate_ propagation of covariances by taking the mean of
        every node pair pixel covariances. This process is computational 
        very expensive and can take a few minutes.
        :py:class:`kite.covariance.Covariance.matrix_focal`

    :param quadtree: Quadtree to work on
    :type quadtree: :py:class:`kite.quadtree.Quadtree`
    :param config: Config object
    :type config: :py:class:`kite.covariance.CovarianceConfig`
    """
    evCovarianceUpdate = Subject()

    def __init__(self, scene, config=CovarianceConfig()):
        self.config = config
        self.frame = scene.frame
        self._quadtree = scene.quadtree
        self._scene = scene
        self._noise_data = None
        self._noise_spectrum_cached = None
        self._initialized = False

        self._log = scene._log.getChild('Covariance')
        self._quadtree.evParamUpdate.subscribe(self._clear)

    def __call__(self, *args, **kwargs):
        return self.getLeafCovariance(*args, **kwargs)

    def _clear(self):
        self.config.a = -9999.
        self.config.b = -9999.
        self.config.variance = -9999.
        self.covariance_matrix = None
        self.covariance_matrix_focal = None
    def noise_data(self, data):
        data = data.copy()
        data = derampMatrix(trimMatrix(data))  # removes nans or 0.
        data[num.isnan(data)] = 0.
        self._noise_data = data
        self._clear()
        self.covarianceUpdate._notify()

    def setNoiseData(self, data):
        self.noise_data = data

    def _mapLeafs(self, nx, ny):
        """Helper function returning appropriate QuadNodes and for maintaining
        the internal mapping

        :param nx: matrix x position
        :type nx: int
        :param ny: matrix y position
        :type ny: int
        :returns: tuple of :py:class:`kite.quadtree.QuadNode` for ``nx``
            and ``ny``
        :rtype: {[tuple]}
        """
        leaf1 = self._quadtree.leafs[nx]
        leaf2 = self._quadtree.leafs[ny]

        self._leaf_mapping[leaf1.id] = nx
        self._leaf_mapping[leaf2.id] = ny

        return leaf1, leaf2

    @property_cached
    def covariance_matrix(self):
        """Covariance matrix calculated from sub-distances of pixels of
        quadtree node pairs.
        """
        return self._calcDistanceMatrix(method='full')

    @property_cached
    def covariance_matrix_focal(self):
        """ Approximate Covariance matrix from quadtree node pair 
        distance only. Fast, use for intermediate steps only and
        finallly use approach `Covariance.matrix` """
        return self._calcDistanceMatrix(method='focal')

    @property_cached
    def weight_matrix(self):
        """ Weight matrix from sqrt{covariance_matrix^-1}
        """
        return num.linalg.inv(self.covariance_matrix)

    @property_cached
    def weight_matrix_focal(self):
        """ Approximated weight matrix from 
        sqrt{covariance_matrix_focal^-1}
        """
        return num.linalg.inv(self.covariance_matrix_focal)

    def _calcDistanceMatrix(self, method='focal', nthreads=0):
        """Construction of the variance-covariance matrix

        :param method: Either ``focal`` point distances are used - this is
            quick but statistically not reliable.
            Or ``full``, where the full quadtree pixel distances matrices are
            calculated
            , defaults to ``focal``
        :type method: str, optional
        :param nthreads: Number of threads to use, ``0`` will use all
            available processors
        :ttype nthreads: int
        :returns: Covariance matrix
        :rtype: {:python:numpy.ndarray}
        """
        self._initialized = True

        nl = len(self._quadtree.leafs)
        self._leaf_mapping = {}

        t0 = time.time()
        ma, mb = self.covariance_model()

        if method == 'focal':
            dist_matrix = num.zeros((nl, nl))
            dist_iter = num.nditer(num.triu_indices_from(dist_matrix))

            for nx, ny in dist_iter:
                leaf1, leaf2 = self._mapLeafs(nx, ny)
                dist = self._leafFocalDistance(leaf1, leaf2)
                dist_matrix[(nx, ny), (ny, nx)] = dist
            cov_matrix = modelCovariance(dist_matrix, ma, mb)

        elif method == 'full':
            leaf_map = num.empty((len(self._quadtree.leafs), 4),
                                 dtype=num.uint32)
            for nl, leaf in enumerate(self._quadtree.leafs):
                leaf, _ = self._mapLeafs(nl, nl)
                leaf_map[nl, 0], leaf_map[nl, 1] = (leaf._slice_rows.start,
                                                    leaf._slice_rows.stop)
                leaf_map[nl, 2], leaf_map[nl, 3] = (leaf._slice_cols.start,
                                                    leaf._slice_cols.stop)
            cov_matrix = covariance_ext.leaf_distances(
                            self._scene.frame.gridE.filled(),
                            self._scene.frame.gridN.filled(),
                            leaf_map, ma, mb, nthreads)
        else:
            raise ValueError('%s method not defined!' % method)

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

    def _getMapping(self, leaf1, leaf2):
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
        """Get the covariances of ``leaf1`` and ``leaf2`` in ``m`` from
        distances.

        :param leaf1: Leaf 1
        :type leaf1: str of `leaf.id` or :py:class:`kite.quadtree.QuadNode`
        :param leaf2: Leaf 2
        :type leaf2: str of `leaf.id` or :py:class:`kite.quadtree.QuadNode`
        :returns: Covariance between ``leaf1`` and ``leaf2``
        :rtype: {float}
        """
        return self.covariance_matrix[self._getMapping(leaf1, leaf2)]

    def getLeafWeight(self, leaf1):
        (nl, _) = self._getMapping(leaf1, leaf1)
        weight_mat = self.weight_matrix_focal
        return num.mean(weight_mat, axis=0)[nl]

    def noiseSpectrum(self, data=None):
        """Calculate the power spectrum from Covariance.noise_data

        :param data: Overwrite Covariance.noise_data, defaults to {None}
        :type data: :py:class:`numpy.ndarray`, optional
        :returns: *(power_spec, k, f_spectrum, kN, kE)*
        :rtype: {tuple}
        """
        if self._noise_spectrum_cached is not None:
            return self._noise_spectrum_cached
        if data is None:
            noise = self.noise_data
        else:
            noise = data.copy()

        f_spec = num.fft.fft2(noise, axes=(0, 1), norm=None)
        f_spec /= noise.size
        f_spec = num.abs(f_spec)

        kE = num.fft.fftfreq(f_spec.shape[1], d=self._quadtree.frame.dE)
        kN = num.fft.fftfreq(f_spec.shape[0], d=self._quadtree.frame.dN)

        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

        k_bin = kN if kN.size > kE.size else kE
        power_spec, k, _ = sp.stats.binned_statistic(k_rad.flatten(),
                                                     f_spec.flatten(),
                                                     statistic='mean',
                                                     bins=k_bin[k_bin > 0])

        self._noise_spectrum_cached = power_spec, k[:-1], f_spec, kN, kE
        return self._noise_spectrum_cached

    def _powerspecFit(self, regime=0):
        """Fitting a function to the power spectrum of noise 
        """
        power_spec, k, _, _, _ = self.noiseSpectrum()

        def selectRegime(k, k1, k2):
            return num.logical_and(k > k1, k < k2)

        regime = selectRegime(k, *noise_regimes[regime])

        try:
            return sp.optimize.curve_fit(modelPowerspec,
                                         k[regime], power_spec[regime],
                                         p0=None, sigma=None,
                                         absolute_sigma=False,
                                         check_finite=True,
                                         bounds=(-num.inf, num.inf),
                                         method=None,
                                         jac=None)
        except RuntimeError:
            self._log.warning('Could not fit the powerspectrum model.')

    def powerspecAnalytical(self, k, regime=0):
        """ Fitting the power spectrum with a given model."""
        p, _ = self._powerspecFit(regime)
        return modelPowerspec(k, *p)

    @staticmethod
    def _powerspecCosineTransform(p_spec, k):
        """Getting the cosine transform of the power spectrum.
        
        The cosine transform of the power spectrum is an estimate
        of the data covariance (see Hanssen, 2001)."""
            p_spec = p_spec[k > 0]
            k = k[k > 0]
            p_spec[num.isnan(p_spec)] = 0.
            cos = sp.fftpack.dct(p_spec, type=2, n=None, norm=None)
            cos *= 2./cos.size

            # Normieren ueber n_k?
            return cos, k

    def covarianceAnalytical(self, regime=0):
        """Estimation of the covariance using a spectral approach.
        
       The covariance of the data is estimated via the noise power spectrum 
       and its cosine transform (Hanssen, 2001). 
        """
        
        _, k, _, kN, kE = self.noiseSpectrum()
        (a, b), _ = self._powerspecFit(regime)
        dk = self._quadtree.frame.dN if kN.size > kE.size\
            else self._quadtree.frame.dE

        spec = modelPowerspec(k, a, b)
        d = num.arange(1, spec.size+1) * dk

        cos, _ = self._powerspecCosineTransform(spec, k)
        return cos, d

    @property
    def covariance_model(self, regime=0):
        """WHy would I fit the cosine trasform again??  """
        if self.config.a == -9999. or self.config.b == -9999:
            cov, d = self.covarianceAnalytical(regime)
            try:
                (a, b), _ =\
                    sp.optimize.curve_fit(modelCovariance, d, cov,
                                          p0=(.001, 1000.))
                self.config.a, self.config.b = (float(a), float(b))
            except RuntimeError:
                self._log.warning('Could not fit the covariance model.')
        return self.config.a, self.config.b

    @property_cached
    def covariance_func(self):
        ''' Covariance function estimated from power spectrum of
        displacement noise patch and its cosine transform.
        '''
        power_spec, k, p_spec, kN, kE = self.noiseSpectrum()
        dk = self._quadtree.frame.dN if kN.size > kE.size\
            else self._quadtree.frame.dE
        d = num.arange(1, power_spec.size+1) * dk
        cov, _ = self._powerspecCosineTransform(power_spec, k)

        return cov, d

    @property_cached
    def structure_func(self):
        """Estimation of the structure function """
        # from http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
        cov, d = self.covariance_func
        power_spec, k, f_spec, kN, kE = self.noiseSpectrum()

        def structure_func(power_spec, d, k):
            struc_func = num.zeros_like(cov)
            for i, d in enumerate(d):
                for ik, tk in enumerate(k):
                    struc_func[i] += (1. - num.cos(tk*d))*power_spec[ik]
                    # struc_func[i] += (1. - num.i0(tk*d))*power_spec[ik]
            # struc_func *= 1./power_spec.size
            return struc_func

        struc_func = structure_func(power_spec, d, k)
        return struc_func, d

    @property
    def variance(self):
        return self.config.variance

    @variance.setter
    def variance(self, value):
        self.config.variance = float(value)

    @variance.getter
    def variance(self):
        if self.config.variance == -9999.:
            self.config.variance = float(num.mean(self.structure_func[0]))
        return self.config.variance

    @property_cached
    def plot(self):
        from kite.plot2d import CovariancePlot
        return CovariancePlot(self)


    @property_cached
    def structure_func(self):
        ''' Structure function derived from ``noise_patch``
        from http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
        '''
        cov, d = self.covariance_func
        power_spec, k, f_spec, kN, kE = self.noiseSpectrum()

        def structure_func(power_spec, d, k):
            struc_func = num.zeros_like(cov)
            for i, d in enumerate(d):
                for ik, tk in enumerate(k):
                    struc_func[i] += (1. - num.cos(tk*d))*power_spec[ik]
                    # struc_func[i] += (1. - num.i0(tk*d))*power_spec[ik]
            # struc_func *= 1./power_spec.size
            return struc_func

        struc_func = structure_func(power_spec, d, k)
        return struc_func, d

    @property
    def variance(self):
        ''' Variance, derived from the mean of
        :py:attr:`kite.covariance.Covariance.structure_func`.
        '''
        return self.config.variance

    @variance.setter
    def variance(self, value):
        self.config.variance = float(value)
        self.evCovarianceUpdate.notify()

    @variance.getter
    def variance(self):
        if self.config.variance == -9999.:
            self.config.variance = float(num.mean(self.structure_func[0]))
        return self.config.variance

    @property_cached
    def plot(self):
        ''' Simple overview plot to summarize the covariance. '''
        from kite.plot2d import CovariancePlot
        return CovariancePlot(self)
