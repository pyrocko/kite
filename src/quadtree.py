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
        return self._scene._mapGridToUTM(*self._focal_point)

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

    def __repr__(self):
        return '''QuadNode:
  llx: %d px
  lly: %d px
  length: %d px
  mean: %.4f
  median: %.4f
  std: %.4f
  var: %.4f
        ''' % (self.llx, self.lly, self.length, self.mean, self.median,
               self.std, self.var)


def createTreeParallel(args):
    base_node, func, epsilon_limit = args
    base_node.createTree(func, epsilon_limit)
    return base_node


class Quadtree(object):
    yaml_tag = '!Quadtree'

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
            '''
            Pathos uses dill instead of pickle, this works w lambdas
            '''

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
        i = 0
        for b in self._base_nodes:
            for n in b.iterTree():
                i += 1
        return i

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


def _covarianceMatrixWorker(args):
    ind, subsampl,\
        node1_utmX, node1_utmY, node2_utmX, node2_utmY = args
    node1_subsmpl = subsampl if not float(node1_utmX.size)/subsampl < 1.\
        else int(num.floor(float(node1_utmX.size)*subsampl))
    node2_subsmpl = subsampl if not float(node2_utmX.size)/subsampl < 1.\
        else int(num.floor(float(node2_utmX.size)*subsampl))

    node1_subsmpl = subsampl
    node2_subsmpl = subsampl
    dx = (node1_utmX.compressed()[::node1_subsmpl][:, num.newaxis] -
          node2_utmX.compressed()[::node2_subsmpl][num.newaxis, :])
    dy = (node1_utmY.compressed()[::node1_subsmpl][:, num.newaxis] -
          node2_utmY.compressed()[::node2_subsmpl][num.newaxis, :])
    d = num.median(num.sqrt(dx**2 + dy**2))
    # cov = self.b * num.exp(-d/self.a)  # * num.cos(d/self.c)
    return ind, d


class Covariance(object):
    """Analytical covariance of quadtree

    The covariance is used as a weighting measure for the optimization process

    :param sc: [description]
    :type sc: [type]
    :param for e in num.linspace(0.1, .00005, num: [description]
    :type for e in num.linspace(0.1, .00005, num: [type]
    """
    def __init__(self, quadtree):
        self._quadtree = quadtree

        self.subsampling = 8
        self.a = 7
        self.b = 1.e-5
        self.c = 1.

        self._log = logging.getLogger('Covariance')

        def clearCovariance():
            self.matrix = None
            self.matrix_fast = None
        self._quadtree.treeUpdate.subscribe(clearCovariance)

    @property_cached
    def matrix(self):
        return self._calcMatrix(method='matrix')

    @property_cached
    def matrix_fast(self):
        return self._calcMatrix(method='focal')

    def _calcMatrix(self, method='focal'):
        nl = len(self._quadtree.leafs)
        cov_matrix = num.zeros((nl, nl))

        if method == 'focal':
            for nx, ny in num.nditer(num.triu_indices_from(cov_matrix)):
                node1 = self._quadtree.leafs[nx]
                node2 = self._quadtree.leafs[ny]

                cov = self._nodeMeanCovariance(node1, node2, method=method)
                cov_matrix[(nx, ny), (ny, nx)] = cov

        elif method == 'matrix':
            from multiprocessing import Pool, cpu_count
            from progressbar import ProgressBar, ETA, Bar
            self._log.info('Building tasks...')
            worker_chunksize = 48 * self.subsampling

            tasks = []
            for nx, ny in num.nditer(num.triu_indices_from(cov_matrix)):
                node1 = self._quadtree.leafs[nx]
                node2 = self._quadtree.leafs[ny]
                tasks.append(((nx, ny), self.subsampling,
                             node1.utm_gridX, node1.utm_gridY,
                             node2.utm_gridX, node2.utm_gridY))
            pbar = ProgressBar(maxval=len(tasks), widgets=[Bar(),
                                                           ETA()])

            self._log.info('Multiprocessing covariance matrix'
                           ' - subsampling %d on %d cpus' %
                           (self.subsampling, cpu_count()))
            pool = Pool(maxtasksperchild=worker_chunksize)
            result = pool.imap_unordered(_covarianceMatrixWorker, tasks,
                                         chunksize=worker_chunksize)
            pool.close()
            pbar.start()
            for i, res in enumerate(result):
                (nx, ny), d = res
                cov_matrix[(nx, ny), (ny, nx)] = d
                pbar.update(i)
            pool.terminate()

        return cov_matrix

    @staticmethod
    def _nodeDistanceFocal(node1, node2):
        return num.sqrt((node1.focal_point_utm[0]
                         - node2.focal_point_utm[0])**2 +
                        (node1.focal_point_utm[1]
                         - node2.focal_point_utm[1])**2)

    def _nodeCovariance(self, node1, node2, method='focal'):
        if method == 'matrix':
            d = self._nodeDistanceMatrix(node1, node2)
        elif method == 'focal':
            d = self._nodeDistanceFocal(node1, node2)
        d *= 1e6
        return d*1e-6
        return self.b * num.exp(-d/self.a)  # * num.cos(d/self.c)

    def _nodeMeanCovariance(self, node1, node2, method='focal'):
        if method == 'matrix':
            return num.mean(self._nodeCovariance(node1, node2, method=method))
        elif method == 'focal':
            return self._nodeCovariance(node1, node2, method=method)

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
