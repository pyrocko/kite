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


def modelPowerspec(k, a, b):
    """Exponential Linear model to estimate a log-linear powerspectrum

    We assume the following log-linear model for the measured powerspectrum

    .. math::

        pow(k) = k^a * \\frac{1}{b}


    :param k: Wavenumber
    :type k: float or :class:`numpy.ndarray`
    :param a: Exponential model factor
    :type a: float
    :param b: Fractional model factor
    :type b: float
    """
    return (k**a)/b


class CovarianceConfig(guts.Object):
    noise_coord = Array.T(
        shape=(None,), dtype=num.float,
        serialize_as='list',
        optional=True,
        help='Noise patch coordinates and size,')
    a = guts.Float.T(
        optional=True,
        help='Exponential covariance model; scaling factor. '
             'See :func:`kite.covariance.modelCovariance`')
    b = guts.Float.T(
        optional=True,
        help='Exponential covariance model; exponential decay. '
             'See :func:`kite.covariance.modelCovariance`')
    variance = guts.Float.T(
        optional=True,
        help='Variance of the model')


class Covariance(object):
    """Analytical covariance for noise estimation of
    :class:`kite.Scene.displacement`.

    The covariance between :attr:`kite.Quadtree.leafs` is
    used as a weighting measure for the optimization process.

    Two different methods are implemented to estimate the covariance function:

    1. The distance between :class:`kite.quadtree.QuadNode`
       leaf focal points, :attr:`kite.Covariance.matrix_focal`
    2. The more *accurate* statistical distances between every nodes pixels,
       this process is computational very expensive and
       can take a few minutes.
       See :class:`kite.Covariance.matrix_focal`.

    :param quadtree: Quadtree to work on
    :type quadtree: :class:`kite.Quadtree`
    :param config: Config object
    :type config: :class:`kite.covariance.CovarianceConfig`
    """
    evChanged = Subject()
    evConfigChanged = Subject()

    def __init__(self, scene, config=CovarianceConfig()):
        self.frame = scene.frame
        self.quadtree = scene.quadtree
        self.scene = scene
        self._noise_data = None
        self._noise_spectrum_cached = None
        self._initialized = False
        self._log = scene._log.getChild('Covariance')

        self.setConfig(config)
        self.quadtree.evChanged.subscribe(self._clear)
        self.scene.evConfigChanged.subscribe(self.setConfig)

    def __call__(self, *args, **kwargs):
        return self.getLeafCovariance(*args, **kwargs)

    def setConfig(self, config=None):
        """ Sets and updated the config of the instance

        :param config: New config instance, defaults to configuration provided
                       by parent :class:`kite.Scene`
        :type config: :class:`kite.covariance.CovarianceConfig`, optional
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
        self.evChanged.notify()

    def _clear(self, config=True):
        if config:
            self.config.a = None
            self.config.b = None
            self.config.variance = None
        self.covariance_matrix = None
        self.covariance_matrix_focal = None
        self.covariance_func = None
        self.weight_matrix = None
        self.weight_matrix_focal = None
        self.structure_func = None
        self._noise_spectrum_cached = None
        self._initialized = False

    @property
    def noise_coord(self):
        """ Coordinates of the noise patch in local coordinates.

        :setter: Set the noise coordinates
        :getter: Get the noise coordinates
        :type: :class:`numpy.array`, ``[llE, llN, sizeE, sizeN]``
        """
        if self.config.noise_coord is None:
            self.noise_data
        return self.config.noise_coord

    @noise_coord.setter
    def noise_coord(self, value):
        self.config.noise_coord = num.array(value)

    @property
    def noise_patch_size_km2(self):
        '''
        :getter: Noise patch size in ``km^2``.
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
        :getter: If the noise data has not been set manually, we grab the
            biggest :class:`kite.quadtree.QuadNode` from
            :class:`kite.Quadtree`.
        :type: :class:`numpy.array`
        '''
        return self._noise_data

    @noise_data.getter
    def noise_data(self):
        if self._noise_data is not None:
            return self._noise_data
        elif self.config.noise_coord is not None:
            print(self.config.noise_coord[:])
            llR, llC = self.scene.frame.mapENMatrix(
                *self.config.noise_coord[:2])
            sR, sC = self.scene.frame.mapENMatrix(
                *self.config.noise_coord[2:])
            slice_rows = slice(llR, llR + sR)
            slice_cols = slice(llC, llC + sC)
            self.noise_data = self.scene.displacement[slice_rows, slice_cols]
        else:
            node = self.getNoiseNode()
            self.noise_data = node.displacement
            self.noise_coord = [node.llE, node.llN,
                                node.sizeE, node.sizeN]
        return self.noise_data

    @noise_data.setter
    def noise_data(self, data):
        data = data.copy()
        data = derampMatrix(trimMatrix(data))  # removes nans or 0.
        data[num.isnan(data)] = 0.
        self._noise_data = data
        self._clear()
        self.evChanged.notify()

    def getNoiseNode(self):
        """ Choose noise node from quadtree """
        t0 = time.time()

        stdmax = max([n.std for n in self.quadtree.nodes])  # noqa
        lmax = max([n.std for n in self.quadtree.nodes])  # noqa

        def costFunction(n):
            nl = num.log2(n.length)/num.log2(lmax)
            ns = n.std/stdmax
            return nl/(1.-ns)

        nodes = sorted(self.quadtree.nodes,
                       key=costFunction)

        self._log.debug('Fetched noise from Quadtree.nodes [%0.8f s]'
                        % (time.time() - t0))
        return nodes[3]

    def _mapLeafs(self, nx, ny):
        """ Helper function returning appropriate
            :class:`kite.quadtree.QuadNode` and for maintaining
            the internal mapping.

        :param nx: matrix x position
        :type nx: int
        :param ny: matrix y position
        :type ny: int
        :returns: tuple of :class:`kite.quadtree.QuadNode` for ``nx``
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
        :type: :class:`numpy.array`,
            size (:class:`kite.Quadtree.nleafs`x:class:`kite.Quadtree.nleafs`)
        """
        return self._calcCovarianceMatrix(method='full')

    @property_cached
    def covariance_matrix_focal(self):
        """ This matrix uses distances between focal points. Fast but
            statistically not reliable method. For final approach use
            :attr:`kite.Covariance.covariance_matrix`.
        :type: :class:`numpy.array`,
            size (:class:`kite.Quadtree.nleafs` x
            :class:`kite.Quadtree.nleafs`)
        """
        return self._calcCovarianceMatrix(method='focal')

    @property_cached
    def weight_matrix(self):
        """ Weight matrix from full covariance :math:`cov^{-1}`.
        :type: :class:`numpy.array`,
            size (:class:`kite.Quadtree.nleafs` x
            :class:`kite.Quadtree.nleafs`)
        """
        return num.linalg.inv(self.covariance_matrix)

    @property_cached
    def weight_matrix_focal(self):
        """ Weight matrix from fast focal method :math:`cov_{focal}^{-1}`.
        :type: :class:`numpy.array`,
            size (:class:`kite.Quadtree.nleafs` x
            :class:`kite.Quadtree.nleafs`)
        """
        return num.linalg.inv(self.covariance_matrix_focal)

    def _calcCovarianceMatrix(self, method='focal', nthreads=0):
        """Calculates the covariance matrix.

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
            cov_matrix = covariance_ext.leaf_distances(
                            self.scene.frame.gridE.filled(),
                            self.scene.frame.gridN.filled(),
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
        """Get the covariance between ``leaf1`` and ``leaf2``.

        :param leaf1: Leaf one
        :type leaf1: str of `leaf.id` or :class:`kite.quadtree.QuadNode`
        :param leaf2: Leaf two
        :type leaf2: str of `leaf.id` or :class:`kite.quadtree.QuadNode`
        :returns: Covariance between ``leaf1`` and ``leaf2``
        :rtype: float
        """
        return self.covariance_matrix[self._getMapping(leaf1, leaf2)]

    def getLeafWeight(self, leaf1):
        (nl, _) = self._getMapping(leaf1, leaf1)
        weight_mat = self.weight_matrix_focal
        return num.mean(weight_mat, axis=0)[nl]

    def noiseSpectrum(self, data=None):
        """Get the noise spectrum from
        :attr:`kite.Covariance.noise_data`.

        :param data: Overwrite Covariance.noise_data, defaults to `None`
        :type data: :class:`numpy.ndarray`, optional
        :returns: (power_spec, k, f_spectrum, kN, kE)
        :rtype: tuple
        """
        if self._noise_spectrum_cached is not None:
            return self._noise_spectrum_cached
        if data is None:
            noise = self.noise_data
        else:
            noise = data.copy()

        f_spec = num.fft.fft2(noise, axes=(0, 1), norm=None)
        f_spec = num.abs(f_spec) / f_spec.size
        # f_spec /= f_spec.size

        kE = num.fft.fftfreq(f_spec.shape[1], d=self.quadtree.frame.dE)
        kN = num.fft.fftfreq(f_spec.shape[0], d=self.quadtree.frame.dN)

        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

        k_bin = kN if kN.size < kE.size else kE
        power_spec, k, _ = sp.stats.binned_statistic(k_rad.flatten(),
                                                     f_spec.flatten(),
                                                     statistic='mean',
                                                     bins=k_bin[k_bin > 0])

        self._noise_spectrum_cached = power_spec, k[:-1], f_spec, kN, kE
        return self._noise_spectrum_cached

    def _powerspecFit(self, regime=0):
        power_spec, k, _, _, _ = self.noiseSpectrum()

        def selectRegime(k, k1, k2):
            return num.logical_and(k > k1, k < k2)

        regime = selectRegime(k, *noise_regimes[regime])

        try:
            return sp.optimize.curve_fit(modelPowerspec,
                                         k[regime], power_spec[regime])
        except RuntimeError:
            self._log.warning('Could not fit the powerspectrum model.')

    def powerspecAnalytical(self, k, regime=0):
        p, _ = self._powerspecFit(regime)
        return modelPowerspec(k, *p)

    def _powerCosineTransform(self, p_spec, k):
            if k.sum() == num.nan:
                self._log.warning('Wavenumber infested with nan values')
            cos = sp.fftpack.dct(p_spec, type=2, norm=None)
            cos /= p_spec.size

            # Normieren ueber n_k?
            return cos, k

    @property_cached
    def covariance_func(self):
        ''' Covariance function derived from displacement noise patch.
        :type: tuple, :class:`numpy.array` (covariance, distance) '''
        power_spec, k, p_spec, kN, kE = self.noiseSpectrum()
        dk = self.quadtree.frame.dN if kN.size < kE.size\
            else self.quadtree.frame.dE

        d = num.arange(1, power_spec.size+1) * dk
        cov, _ = self._powerCosineTransform(power_spec, k)

        return cov, d

    def covarianceAnalytical(self, regime=0):
        ''' Analytical covariance based on the spectral model fit
        from :attr:`kite.covariance.modelPowerspec`

        :return: Covariance and corresponding distances.
        :rtype: tuple, :class:`numpy.array` (covariance_analytical, distance)
        '''
        _, k, _, kN, kE = self.noiseSpectrum()
        (a, b), _ = self._powerspecFit(regime)
        dk = self.quadtree.frame.dN if kN.size < kE.size\
            else self.quadtree.frame.dE

        spec = modelPowerspec(k, a, b)
        d = num.arange(1, spec.size+1) * dk

        cos, _ = self._powerCosineTransform(spec, k)
        return cos, d

    @property
    def covariance_model(self, regime=0):
        if self.config.a is None or self.config.b is None:
            cov, d = self.covarianceAnalytical(regime)
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
        :getter: RMS missfit between :class:`kite.Covariance.covariance_model`
            and :class:`kite.Covariance.covariance_func`
        :type: float
        '''
        cov, d = self.covariance_func
        cov_mod = modelCovariance(d, *self.covariance_model)
        return num.sqrt(num.mean(cov**2 - cov_mod**2))

    @property_cached
    def structure_func(self):
        ''' Structure function derived from ``noise_patch``
        :type: tuple, :class:`numpy.array` (structure_func, distance)

        Adapted from
        http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
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
        :attr:`kite.Covariance.structure_func`.

        :setter: Set the variance manually
        :getter: Retrieve the variance
        :type: float
        '''
        return self.config.variance

    @variance.setter
    def variance(self, value):
        self.config.variance = float(value)
        self.evChanged.notify()

    @variance.getter
    def variance(self):
        if self.config.variance is None:
            self.config.variance = float(num.mean(self.structure_func[0]))
        return self.config.variance

    def export_weight_matrix(self, filename):
        """ Export the :attr:`kite.Covariance.weight_matrix` through numpy.
        The data can be loaded through :fund:`numpy.loadtxt`.

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
