import numpy as num
import logging
from pyrocko import guts
from kite.meta import Subject, property_cached
from quadtree import QuadNode


def _leafMatrixCovarianceWorker(args):
    """Worker function serving :python:`multiprocessing.Pool`

    :param args:
        `(ind, subsampl, leaf1_utmX, leaf1_utmY, leaf2_utmX, leaf2_utmY)`
        Where `ind` is tuple of matrix indices `(nx, ny)`, `subsampl`
        subsampling factor `leaf?_utm?` are 2-dim masekd arrays holding UTM
        data from :python:`kite.quadtree.QuadNode`.
    :type args: [tuple]
    :returns: ((nx, ny), covariance)
    :rtype: {[tuple]}
    """
    ind, subsampl,\
        leaf1_utmX, leaf1_utmY, leaf2_utmX, leaf2_utmY = args
    leaf1_subsmpl = subsampl if not float(leaf1_utmX.size)/subsampl < 1.\
        else int(num.floor(float(leaf1_utmX.size)*subsampl))
    leaf2_subsmpl = subsampl if not float(leaf2_utmX.size)/subsampl < 1.\
        else int(num.floor(float(leaf2_utmX.size)*subsampl))

    leaf1_subsmpl = subsampl
    leaf2_subsmpl = subsampl
    # Looks ugly but re we want to conserve memory
    d = num.median(num.sqrt(
        (leaf1_utmX.compressed()[::leaf1_subsmpl][:, num.newaxis] -
         leaf2_utmX.compressed()[::leaf2_subsmpl][num.newaxis, :])**2 +
        (leaf1_utmY.compressed()[::leaf1_subsmpl][:, num.newaxis] -
         leaf2_utmY.compressed()[::leaf2_subsmpl][num.newaxis, :])**2))
    cov = d
    # cov = self.b * num.exp(-d/self.a)  # * num.cos(d/self.c)
    return ind, cov


class Covariance(guts.Object):
    """Analytical covariance used for weighting of quadtree.

    The covariance between :python:`kite.quadtree.Quadtree` leafs is used as a
    weighting measure for the optimization process.

    We assume the analytical formula
        `cov(dist) = c * exp(-dist/b) [* cos(dist/a)]`

    where `dist` is
    1) the distance between quadleaf focal points (`Covariance.matrix_focal`)
    2) statistical distances between quadleaf pixels to pixel
        (`Covariance.matrix`), subsampled accoring to `Covariance.subsampling`.

    :param quadtree: Quadtree to work on
    :type quadtree: `:python:kite.quadtree.Quadtree`
    :param a: scaling the cosinus term. `None` disabled this part of the term,
        defaults to None.
    :type a: number, optional
    :param b: [description], defaults to 1.
    :type b: number, optional
    :param c: [description], defaults to 1.
    :type c: number, optional
    :param subsampling: Subsampling of distances, defaults to 8
    :type subsampling: number, optional
    """
    a = guts.Float.T(default=1.,
                     help='Weight factor a')
    b = guts.Float.T(default=1.,
                     help='Weight factor b')
    c = guts.Float.T(default=1.,
                     help='Weight factor c')
    subsampling = guts.Int(default=8,
                           help='Subsampling of distance matrices')

    def __init__(self, quadtree, a=None, b=1., c=1., subsampling=8):
        self._quadtree = quadtree

        self.a = a
        self.b = b
        self.c = c
        self.subsampling = subsampling

        self._leaf_mapping = None

        self.covarianceUpdate = Subject()

        def clearCovariance():
            self.matrix = None
            self.matrix_focal_points = None
        self._quadtree.treeUpdate.subscribe(clearCovariance)
        self.covarianceUpdate.subscribe(clearCovariance)

        self._log = logging.getLogger('Covariance')

    def __call__(self, *args, **kwargs):
        return self.getWeight(*args, **kwargs)

    @property_cached
    def matrix(self):
        """ Covariance matrix calculated from sub-distances pairs from quadtree
        node-to-node, subsampled by `Covariance.subsampling`
        """
        return self._calcMatrix(method='matrix')

    @property_cached
    def matrix_focal(self):
        """ This matrix uses distances between focal points. Fast but
        statistically not reliable method. For final approach use
        `Covariance.matrix` """
        return self._calcMatrix(method='focal')

    def _getLeafs(self, nx, ny):
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
        calculated, subsampled as set in `Covariance.subsampling`.
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
                leaf1, leaf2 = self._getLeafs(nx, ny)

                cov = self._leafFocalCovariance(leaf1, leaf2)
                cov_matrix[(nx, ny), (ny, nx)] = cov

        elif method == 'matrix':
            from multiprocessing import Pool, cpu_count
            from progressbar import ProgressBar, ETA, Bar

            self._log.info('Preprocessing covariance matrix'
                           ' - subsampling %dx on %d cpus' %
                           (self.subsampling, cpu_count()))
            worker_chunksize = 48 * self.subsampling

            tasks = []
            for nx, ny in cov_iter:
                leaf1, leaf2 = self._getLeafs(nx, ny)

                tasks.append(((nx, ny), self.subsampling,
                             leaf1.utm_grid_x, leaf1.utm_grid_y,
                             leaf2.utm_grid_x, leaf2.utm_grid_y))

            pool = Pool(maxtasksperchild=worker_chunksize)
            results = pool.imap_unordered(_leafMatrixCovarianceWorker, tasks,
                                          chunksize=worker_chunksize)
            pool.close()

            pbar = ProgressBar(maxval=len(tasks), widgets=[Bar(),
                                                           ETA()]).start()
            for i, result in enumerate(results):
                (nx, ny), d = result
                cov_matrix[(nx, ny), (ny, nx)] = d
                pbar.update(i)
            print "Joining pool..."
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
        if isinstance(leaf1, QuadNode):
            leaf1 = leaf1.id
        if isinstance(leaf2, QuadNode):
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

__all__ = ['Covariance']
