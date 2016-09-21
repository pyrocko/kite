import numpy as num
import logging
import time
from kite.meta import Subject, property_cached
# from pyrock.util import clock


class QuadNode(object):
    """A Node in the Quadtree
    """

    def __init__(self, tree, llx, lly, length):
        self.llx = int(llx)
        self.lly = int(lly)
        self.length = int(length)

        self._tree = tree
        self._scene = self._tree._scene
        self._slice_x = slice(self.llx, self.llx+self.length)
        self._slice_y = slice(self.lly, self.lly+self.length)

        self.id = 'node_%d-%d-%d' % (self.llx, self.lly, self.length)
        self.children = None

    @property_cached
    def nan_fraction(self):
        return float(num.sum(num.isnan(self.data)))/self.data.size

    @property_cached
    def mean(self):
        return num.nanmean(self.data)

    @property_cached
    def median(self):
        return num.nanmedian(self.data)

    @property_cached
    def std(self):
        return num.nanstd(self.data)

    @property_cached
    def var(self):
        return num.nanvar(self.data)

    @property_cached
    def median_std(self):
        '''Standard deviation from median'''
        return num.nanstd(self.data - self.median)

    @property_cached
    def mean_std(self):
        '''Standard deviation from mean'''
        return num.nanstd(self.data - self.mean)

    @property_cached
    def _focal_point(self):
        w_x = num.linspace(0, 1., self.data.shape[0], endpoint=True)
        w_y = num.linspace(0, 1., self.data.shape[1], endpoint=True)
        w_X, w_Y = num.meshgrid(w_x, w_y, sparse=False, copy=False)

        nan = num.isnan(self.data)
        x = num.median(w_X.T[~nan])*self.data.shape[0] + self.llx
        y = num.median(w_Y.T[~nan])*self.data.shape[1] + self.lly
        return x, y

    @property_cached
    def focal_point_utm(self):
        # return self._scene._mapGridToUTM(*self._focal_point)
        x = num.median(self.utm_gridX.compressed())
        y = num.median(self.utm_gridY.compressed())
        return x, y

    @property_cached
    def bilinear_std(self):
        raise NotImplementedError('Bilinear fit not implemented')

    @property_cached
    def data(self):
        return self._scene.displacement[self._slice_x, self._slice_y]

    @property_cached
    def utm_gridX(self):
        return self._scene.utm_gridX[self._slice_x, self._slice_y]

    @property_cached
    def utm_gridY(self):
        return self._scene.utm_gridY[self._slice_x, self._slice_y]

    def iterTree(self):
        yield self
        if self.children is not None:
            for c in self.children:
                for rc in c.iterTree():
                    yield rc

    def iterLeafs(self):
        if self.children is None:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafs():
                    yield q

    def iterLeafsEval(self):
        if (self._tree._split_func(self) < self._tree.epsilon and
            not self.length > self._tree._tile_size_lim_p[1])\
           or self.children is None:
            yield self
        elif self.children[0].length < self._tree._tile_size_lim_p[0]:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafsEval():
                    yield q

    def _iterSplitNode(self):
        if self.length == 1:
            yield None
        for _nx, _ny in ((0, 0), (0, 1), (1, 0), (1, 1)):
            _q = QuadNode(self._tree,
                          self.llx + self.length/2 * _nx,
                          self.lly + self.length/2 * _ny,
                          self.length/2)
            if _q.data.size == 0 or num.isnan(_q.data).all():
                continue
            yield _q

    def createTree(self, eval_func, epsilon_limit):
        if eval_func(self) > epsilon_limit:  # or\
            # self.length > .1 * max(self._tree._data.shape): !! Very Expensive
            self.children = [c for c in self._iterSplitNode()]
            for c in self.children:
                c.createTree(eval_func, epsilon_limit)
        else:
            self.children = None

    def __getstate__(self):
        return self.llx, self.lly, self.length,\
               self.children, self._tree

    def __setstate__(self, state):
        self.llx, self.lly, self.length,\
            self.children, self._tree = state


def createTreeParallel(args):
    base_node, func, epsilon_limit = args
    base_node.createTree(func, epsilon_limit)
    return base_node


class Quadtree(object):
    """Quadtree for simplifying InSAR displacement data held in
    :python:`kite.Scene`

    Post-earthquake InSAR displacement scenes can hold a vast amount of data,
    which is unsuiteable for use with modelling code. By simplifying the data
    systematically through a parametrized quadtree we can reduce the dataset to
    significant displacements and have high-resolution where it matters and
    lower resolution at regions with less or constant deformation.

    :param epsilon: %0.3f: [description]
    :type epsilon: %0.3f: [type]
    :param epsilon_init: %0.3f: [description]
    :type epsilon_init: %0.3f: [type]
    :param epsilon_limit: %0.3f: [description]
    :type epsilon_limit: %0.3f: [type]
    :param nleafs: %d: [description]
    :type nleafs: %d: [type]
    :param split_method: %s: [description]
    :type split_method: %s: [type]
    :param
    """
    def __init__(self, scene, epsilon=None):
        self._split_methods = {
            'mean_std': ['Std around mean', lambda node: node.mean_std],
            'median_std': ['Std around median', lambda node: node.median_std],
            'std': ['Standard deviation (std)', lambda node: node.std],
        }
        self._norm_methods = {
            'mean': lambda node: node.mean,
            'median': lambda node: node.median,
        }

        self._scene = scene
        self._data = self._scene.displacement

        self._epsilon = None
        self._nan_allowed = None
        self._tile_size_lim = (250, 5000)

        self._leafs = None

        self._log = logging.getLogger('Quadtree')

        self.splitMethodChanged = Subject()
        self.treeUpdate = Subject()

        self.setSplitMethod('median_std')

    def setSplitMethod(self, split_method, parallel=False):
        """Set splitting method for quadtree tiles

        * `mean_std` tiles standard deviation from tile's mean is evaluated
        * `median_std` tiles standard deviation from tile's median is evaluated
        * `std` tiles standard deviation is evaluated

        :param split_method: Choose from methods
                             `['mean_std', 'median_std', 'std']`
        :type split_method: string
        :raises: AttributeError
        """
        if split_method not in self._split_methods.keys():
            raise AttributeError('Method %s not in %s'
                                 % (split_method, self._split_methods))

        self.split_method = split_method
        self._split_func = self._split_methods[split_method][1]

        # Clearing cached properties through None
        self._epsilon_init = None
        self._epsilon_limit = None
        self.epsilon = self._epsilon_init

        self._initTree(parallel)
        self.splitMethodChanged._notify()

    def _initTree(self, parallel):
        t0 = time.time()
        if parallel:
            from pathos.pools import ProcessPool as Pool
            # Pathos uses dill instead of pickle, this works w lambdas

            pool = Pool()
            self._log.info('Utilizing %d cpu cores' % pool.nodes)
            res = pool.map(createTreeParallel, [(b,
                                                 self._split_func,
                                                 self._epsilon_limit)
                                                for b in self._base_nodes])
            self._base_nodes = [r for r in res]

        else:
            for b in self._base_nodes:
                b.createTree(self._split_func, self._epsilon_limit)

        self._log.info('Tree created, %d nodes [%0.8f s]' % (self.nnodes,
                                                             time.time()-t0))

    @property
    def epsilon(self):
        return self._epsilon

    @property_cached
    def _epsilon_init(self):
        return num.nanstd(self._data)
        # return num.mean([self._split_func(b) for b in self._base_nodes])

    @property_cached
    def _epsilon_limit(self):
        return self._epsilon_init * .2

    @epsilon.setter
    def epsilon(self, value):
        value = float(value)
        if self._epsilon == value:
            return
        if value < self._epsilon_limit:
            self._log.info(
                'Epsilon is out of bounds [%0.3f], epsilon_limit %0.3f' %
                (value, self._epsilon_limit))
            return
        self.leafs = None
        self._epsilon = value

        self.treeUpdate._notify()
        return

    @property
    def nan_allowed(self):
        return self._nan_allowed

    @nan_allowed.setter
    def nan_allowed(self, value):
        if value > 1. or value < 0.:
            raise AttributeError('NaN fraction must be 0 <= nan_allowed <=1 ')
        if value == 1.:
            value = None

        self.leafs = None
        self._nan_allowed = value
        self.treeUpdate._notify()

    @property
    def tile_size_lim(self):
        return self._tile_size_lim

    @tile_size_lim.setter
    def tile_size_lim(self, value):
        tile_size_min, tile_size_max = value
        if tile_size_min > tile_size_max:
            self._log.info('tile_size_min > tile_size_max is required')
            return
        self._tile_size_lim = (tile_size_min, tile_size_max)

        self._tile_size_lim_p = None
        self.leafs = None
        self.treeUpdate._notify()

    @property_cached
    def _tile_size_lim_p(self):
        dp = self._scene.UTMExtent()[-1]
        return (int(self.tile_size_lim[0] / dp),
                int(self.tile_size_lim[1] / dp))

    @property
    def nnodes(self):
        nnodes = 0
        for b in self._base_nodes:
            for n in b.iterTree():
                nnodes += 1
        return nnodes

    @property_cached
    def leafs(self):
        t0 = time.time()
        leafs = []
        for b in self._base_nodes:
            leafs.extend([l for l in b.iterLeafsEval()])
        if self.nan_allowed is not None:
            leafs[:] = [l for l in leafs if l.nan_fraction < self.nan_allowed]
        self._log.info('Gathering leafs (%d) for epsilon %.4f [%0.8f s]' %
                       (len(leafs), self.epsilon, time.time()-t0))
        return leafs

    @property
    def leaf_means(self):
        return num.array([l.mean for l in self.leafs])

    @property
    def leaf_medians(self):
        return num.array([l.median for l in self.leafs])

    @property
    def _leaf_focal_points(self):
        return num.array([l._focal_point for l in self.leafs])

    @property
    def leaf_focal_points_utm(self):
        return num.array([l.focal_point_utm for l in self.leafs])

    @property
    def leaf_matrix_means(self):
        return self._getLeafsNormMatrix(method='mean')

    @property
    def leaf_matrix_medians(self):
        return self._getLeafsNormMatrix(method='median')

    def _getLeafsNormMatrix(self, method='median'):
        if method not in self._norm_methods.keys():
            raise AttributeError('Method %s is not in %s' % (method,
                                 self._norm_methods.keys()))

        leaf_matrix = num.empty_like(self._data)
        leaf_matrix.fill(num.nan)
        for l in self.leafs:
            leaf_matrix[l.llx:l.llx+l.length, l.lly:l.lly+l.length] = \
                self._norm_methods[method](l)
        return leaf_matrix

    @property_cached
    def _base_nodes(self):
        self._base_nodes = []
        init_length = num.power(2,
                                num.ceil(num.log(num.min(self._data.shape)) /
                                         num.log(2)))/4
        nx, ny = num.ceil(num.array(self._data.shape)/init_length)

        for ix in xrange(int(nx)):
            for iy in xrange(int(ny)):
                llx = ix * init_length
                lly = iy * init_length
                self._base_nodes.append(QuadNode(self, llx, lly, init_length))

        if len(self._base_nodes) == 0:
            raise AssertionError('Could not init base nodes.')
        return self._base_nodes

    def UTMExtent(self):
        return self._scene.UTMExtent()

    @property_cached
    def plot(self):
        from kite.plot2d import PlotQuadTree2D
        return PlotQuadTree2D(self)

    @property_cached
    def covariance(self):
        return Covariance(self)

    def getStaticTarget(self):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError

    def __str__(self):
        return '''
Quadtree for %s
  initiated: %s
  epsilon: %0.3f
  epsilon_init: %0.3f
  epsilon_limit: %0.3f
  nleafs: %d
  split_method: %s
        ''' % (repr(self._scene), (self._base_nodes is not None),
               self.epsilon, self._epsilon_init, self._epsilon_limit,
               len(self.leafs), self.split_method)


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


class Covariance(object):
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
                             leaf1.utm_gridX, leaf1.utm_gridY,
                             leaf2.utm_gridX, leaf2.utm_gridY))

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

__all__ = '''
Quadtree
Covariance
'''.split()


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss(2000, 2000)

    for e in num.linspace(0.1, .00005, num=30):
        sc.quadtree.epsilon = e
    # qp = Plot2DQuadTree(qt, cmap='spectral')
    # qp.plot()
