import numpy as num
import scipy as sp

import logging
from pyrocko import guts
from kite.meta import Subject, property_cached, trimMatrix, derampMatrix

from multiprocessing import Pool, cpu_count
from progressbar import ProgressBar, ETA, Bar

__all__ = ['Covariance', 'CovarianceConfig']


class CovarianceConfig(guts.Object):
    a = guts.Float.T(default=1.,
                     help='Weight factor a - cosine decay')
    b = guts.Float.T(default=1.,
                     help='Weight factor b - exponential decay')
    c = guts.Float.T(default=1.,
                     help='Weight factor c - covariance scaling')
    variance = guts.Float.T(default=9999.,
                            help='Node variance')
    distance_cutoff = guts.Int.T(default=35e3,
                                 help='Cutoff distance for covariance weight '
                                      'matrix -> cov(d>distance_cutoff)=0')
    subsampling = guts.Int.T(default=8,
                             help='Subsampling of distance matrices')


def deramp(data):
    """ Deramp through fitting a bilinear plane
    """
    mx = num.nanmean(data, axis=0)
    my = num.nanmean(data, axis=1)
    cx = num.nanmean(mx)
    cy = num.nanmean(my)
    mx -= cx
    my -= cy
    mx[num.isnan(mx)] = 0.
    my[num.isnan(my)] = 0.

    ix = num.arange(mx.size)
    iy = num.arange(my.size)
    dx, _, _, _, _ = sp.stats.linregress(ix, mx)
    dy, _, _, _, _ = sp.stats.linregress(iy, my)

    rx = (ix * dx + cx)
    ry = (iy * dy + cy)

    return data - rx[num.newaxis, :] - ry[:, num.newaxis]


def _workerLeafMatrixCovariance(args):
    """Worker function serving :python:`multiprocessing.Pool`

    :param args:
        `(ind, subsampl, leaf1_utm_grid_x, leaf1_utm_grid_y,
          leaf2_utm_grid_x, leaf2_utm_grid_y)`
        Where `ind` is tuple of matrix indices `(nx, ny)`, `subsampl`
        subsampling factor `leaf?_utm?` are 2-dim masekd arrays holding UTM
        data from :python:`kite.quadtree.QuadNode`.
    :type args: [tuple]
    :returns: ((nx, ny), covariance)
    :rtype: {[tuple]}
    """
    (ind, subsampl,
     leaf1_utm_grid_x, leaf1_utm_grid_y,
     leaf2_utm_grid_x, leaf2_utm_grid_y) = args
    leaf1_subsmpl = subsampl if not float(leaf1_utm_grid_x.size)/subsampl < 1.\
        else int(num.floor(float(leaf1_utm_grid_x.size)*subsampl))
    leaf2_subsmpl = subsampl if not float(leaf2_utm_grid_x.size)/subsampl < 1.\
        else int(num.floor(float(leaf2_utm_grid_x.size)*subsampl))

    # Looks ugly but re we want to conserve memory
    d = num.median(num.sqrt(
        (leaf1_utm_grid_x.compressed()[::leaf1_subsmpl][:, num.newaxis] -
         leaf2_utm_grid_x.compressed()[::leaf2_subsmpl][num.newaxis, :])**2 +
        (leaf1_utm_grid_y.compressed()[::leaf1_subsmpl][:, num.newaxis] -
         leaf2_utm_grid_y.compressed()[::leaf2_subsmpl][num.newaxis, :])**2))
    cov = d
    # cov = self.b * num.exp(-d/self.a)  # * num.cos(d/self.c)
    return ind, cov


def _workerCovariance(args):
    (subsmpl, d_cutoff,
     node1_data, node1_utm_grid_x, node1_utm_grid_y,
     node2_data, node2_utm_grid_x, node2_utm_grid_y) = args

    # n1_tu = num.triu_indices_from(node1_utm_grid_x)
    # n2_tu = num.triu_indices_from(node2_utm_grid_x)

    d = num.sqrt(
        (node1_utm_grid_x.compressed()[::subsmpl][:, num.newaxis] -
         node2_utm_grid_x.compressed()[::subsmpl][num.newaxis, :])**2 +
        (node1_utm_grid_y.compressed()[::subsmpl][:, num.newaxis] -
         node2_utm_grid_y.compressed()[::subsmpl][num.newaxis, :])**2)
    # ind = num.triu_indices_from(d, k=0)

    d = d
    if d.min() > d_cutoff:
        pass
        # return None, None, None

    # node1_data = sps.detrend(node1_data, type='linear', axis=0)
    # node1_data = sps.detrend(node1_data, type='linear', axis=1)

    # node2_data = sps.detrend(node2_data, type='linear', axis=0)
    # node2_data = sps.detrend(node2_data, type='linear', axis=1)

    node1_data = deramp(node1_data)
    node2_data = deramp(node2_data)
    node1_data = node1_data.compressed()[::subsmpl]
    node2_data = node2_data.compressed()[::subsmpl]

    var = num.sqrt(num.abs(node1_data[:, num.newaxis] -
                           node2_data[num.newaxis, :]))
    cov = node1_data[:, num.newaxis] * node2_data[num.newaxis, :]
    print cov.shape
    cov = cov

    var = var[d <= d_cutoff]
    cov = cov[d <= d_cutoff]
    d = d[d <= d_cutoff]
    '''
    Binning the results
    '''
    def meanVariance(data):
        # Formular for variance
        # (mean(sqrt(|d1-d2|))**4 / (.457+.0494/nsamples_p_bin))/2
        return (num.mean(data)**4 / (.457 + .494/data.size)) / 2

    def meanCovariance(data):
        return 1./(2*data.size) * num.nansum(data)

    nbins = 500
    var, xedg, _ = sp.stats.binned_statistic(d, var,
                                             statistic=meanVariance,
                                             bins=nbins)
    cov, xedg, _ = sp.stats.binned_statistic(d, cov,
                                             statistic='mean',
                                             bins=nbins)
    dist = xedg[:-1] + (xedg[1]-xedg[2])/2
    return dist, var, cov


def covarianceFunction(distance, a, b):
        return b * num.exp(-distance/a)


class Covariance(guts.Object):
    """Analytical covariance used for weighting of quadtree.

    The covariance between :python:`kite.quadtree.Quadtree` leafs is used as a
    weighting measure for the optimization process.

    We assume the analytical formula
        `cov(dist) = c * exp(-dist/b) [* cos(dist/a)]`

    where `dist` is
    1) the distance between quadleaf focal points (`Covariance.matrix_focal`)
    2) statistical distances between quadleaf pixels to pixel
        (`Covariance.matrix`), subsampled accoring to
        `Covariance.config.subsampling`.

    :param quadtree: Quadtree to work on
    :type quadtree: `:python:kite.quadtree.Quadtree`
   """
    def __init__(self, quadtree, config=CovarianceConfig()):
        self.covarianceUpdate = Subject()
        self.covarianceUpdate.subscribe(self._clearCovariance)

        self.config = config
        self._quadtree = quadtree
        self._noise_data = None

        self._log = logging.getLogger('Covariance')

    def __call__(self, *args, **kwargs):
        return self.getDistance(*args, **kwargs)

    def _clearCovariance(self):
        self.matrix = None
        self.matrix_focal_points = None
        self.covariance_func = None

    @property
    def noise_data(self, data):
        return self._noise_data

    @noise_data.getter
    def noise_data(self):
        if self._noise_data is not None:
            return self._noise_data
        nodes = sorted(self._quadtree.leafs,
                       key=lambda n: n.length/(n.nan_fraction+1))
        self._noise_data = nodes[-1].data
        return self.noise_data

    @noise_data.setter
    def noise_data(self, data):
        self._noise_data = data

    def setNoiseData(self, data):
        self.noise_data = data

    @property
    def subsampling(self):
        return self.config.subsampling

    @subsampling.setter
    def subsampling(self, value):
        self._clearCovariance()
        self.config.subsampling = value

    @property_cached
    def matrix(self):
        """ Covariance matrix calculated from sub-distances pairs from quadtree
        node-to-node, subsampled by `Covariance.config.subsampling`
        """
        return self._calcMatrix(method='matrix')

    @property_cached
    def matrix_focal(self):
        """ This matrix uses distances between focal points. Fast but
        statistically not reliable method. For final approach use
        `Covariance.matrix` """
        return self._calcMatrix(method='focal')

    def _mapLeafs(self, nx, ny):
        """Helper function returning appropriate QuadNodes and for maintaining
        the internal mapping

        :param nx: matrix x position
        :type nx: int
        :param ny: matrix y position
        :type ny: int
        :returns: tuple of `:python:kite.quadtree.QuadNode` for nx and ny
        :rtype: {[tuple]}
        """
        leaf1 = self._quadtree.leafs[nx]
        leaf2 = self._quadtree.leafs[ny]

        self._leaf_mapping[leaf1.id] = nx
        self._leaf_mapping[leaf2.id] = ny

        return leaf1, leaf2

    def _calcMatrix(self, method='focal'):
        """Calculates the covariance matrix

        :param method: Either `'focal'` point distances are used - this is
        quick but statistically not reliable.
        Or `'matrix'`, where the full quadtree pixel distances matrices are
        calculated, subsampled as set in `Covariance.config.subsampling`.
        , defaults to 'focal'
        :type method: str, optional
        :returns: Covariance matrix
        :rtype: {:python:numpy.ndarray}
        """
        nl = len(self._quadtree.leafs)
        cov_matrix = num.zeros((nl, nl))
        cov_iter = num.nditer(num.triu_indices_from(cov_matrix))

        self._leaf_mapping = {}

        if method == 'focal':
            for nx, ny in cov_iter:
                leaf1, leaf2 = self._mapLeafs(nx, ny)

                cov = self._leafFocalCovariance(leaf1, leaf2)
                cov_matrix[(nx, ny), (ny, nx)] = cov

        elif method == 'matrix':
            self._log.info('Preprocessing covariance matrix'
                           ' - subsampling %dx on %d cpus...' %
                           (self.config.subsampling, cpu_count()))
            worker_chunksize = 24 * self.config.subsampling

            tasks = []
            for nx, ny in cov_iter:
                leaf1, leaf2 = self._mapLeafs(nx, ny)

                tasks.append(((nx, ny), self.config.subsampling,
                             leaf1.utm_grid_x, leaf1.utm_grid_y,
                             leaf2.utm_grid_x, leaf2.utm_grid_y))
            pool = Pool(maxtasksperchild=worker_chunksize)
            results = pool.imap_unordered(_workerLeafMatrixCovariance, tasks,
                                          chunksize=1)
            pool.close()

            pbar = ProgressBar(maxval=len(tasks), widgets=[Bar(),
                                                           ETA()]).start()
            for i, result in enumerate(results):
                (nx, ny), cov = result
                cov_matrix[(nx, ny), (ny, nx)] = cov
                pbar.update(i+1)
            pool.join()

        return cov_matrix

    @staticmethod
    def _leafFocalDistance(leaf1, leaf2):
        return num.sqrt((leaf1.focal_point_utm[0]
                         - leaf2.focal_point_utm[0])**2 +
                        (leaf1.focal_point_utm[1]
                         - leaf2.focal_point_utm[1])**2)

    def _leafFocalCovariance(self, leaf1, leaf2):
        d = self._leafFocalDistance(leaf1, leaf2)
        return d
        return self.b * num.exp(-d/self.a)  # * num.cos(d/self.c)

    def _getMapping(self, leaf1, leaf2):
        if not isinstance(leaf1, str):
            leaf1 = leaf1.id
        if not isinstance(leaf2, str):
            leaf2 = leaf2.id
        try:
            return self._leaf_mapping[leaf1], self._leaf_mapping[leaf2]
        except KeyError as e:
            raise KeyError('Unknown quadtree leaf with id %s' % e)

    def getDistance(self, leaf1, leaf2):
        """Get the distances between `leaf1` and `leaf2` in `m`

        :param leaf1: Leaf 1
        :type leaf1: str of `leaf.id` or :python:`kite.quadtree.QuadNode`
        :param leaf2: Leaf 2
        :type leaf2: str of `leaf.id` or :python:`kite.quadtree.QuadNode`
        :returns: Distance between `leaf1` and `leaf2`
        :rtype: {float}
        """
        return self.matrix[self._getMapping(leaf1, leaf2)]

    def powerspecNoise(self, data=None):
        if data is None:
            noise = self.noise_data.copy()
        else:
            noise = data.copy()

        noise = trimMatrix(noise)  # removes nans or 0.
        noise = derampMatrix(noise)
        noise[num.isnan(noise)] = 0.

        f_spec = num.fft.fft2(noise, axes=(0, 1), norm=None)
        p_spec = num.abs(f_spec)  # / f_spec.size

        p_spec_x = num.mean(p_spec, axis=0)
        p_spec_y = num.mean(p_spec, axis=1)

        k_x = num.fft.fftfreq(p_spec_x.size, d=self._quadtree.utm.dx)
        k_y = num.fft.fftfreq(p_spec_y.size, d=self._quadtree.utm.dy)

        return p_spec_x, k_x, p_spec_y, k_y

    def covariance_analytical(self, distance):
        '''Retrieve analytical covariance fitted from
        Covariance.covariance_func
        '''
        return covarianceFunction(distance, *self.covariance_coeff)

    @property_cached
    def covariance_coeff(self):
        weights = num.linspace(0., 1., self.covariance_func[1].size)
        p, cov = sp.optimize.curve_fit(covarianceFunction,
                                       self.covariance_func[1],
                                       self.covariance_func[0],
                                       p0=None,
                                       sigma=weights,
                                       check_finite=True,
                                       method=None, jac=None)
        return p

    @property_cached
    def covariance_func(self):
        ''' Covariance function derived from displacement noise patch
        '''
        def covarianceCosine(p_spec, w):
            p_spec = p_spec[w > 0]
            w = w[w > 0]
            p_spec[num.isnan(p_spec)] = 0.
            cos = sp.fftpack.dct(p_spec, type=2, n=None, norm='ortho')
            return cos, w

        ps_x, k_x, ps_y, k_y = self.powerspecNoise()

        cov_x, _ = covarianceCosine(ps_x, k_x)
        cov_y, _ = covarianceCosine(ps_y, k_y)

        d_x = num.arange(1, cov_x.size+1) * self._quadtree.utm.dx
        d_y = num.arange(1, cov_y.size+1) * self._quadtree.utm.dy

        return cov_x, d_x, cov_y, d_y

    @property_cached
    def structure_func(self):
        # from http://clouds.eos.ubc.ca/~phil/courses/atsc500/docs/strfun.pdf
        cov_x, d_x, cov_y, d_y = self.covariance_func
        p_spec_x, k_x, p_spec_y, k_y = self.powerspecNoise()

        struc_func = num.zeros_like(cov_x)
        for i, d in enumerate(d_x):
            for ik, k in enumerate(k_x[k_x>0.]):
                struc_func[i] += 1. - num.cos(-k*d)*cov_x[ik]
        return struc_func, d_x
