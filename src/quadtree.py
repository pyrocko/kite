import numpy as num
import logging
import time
from kite.meta import Subject, property_cached


class QuadNode(object):
    """A Node in the Quadtree
    """
    # __slots__ = ('llx', 'lly', 'length', '_data_complete', 'children'
    #              '_cached_nan_fraction', '_cached_mean', '_cached_ median',
    #              '_cached_std', '_cached_var', '_cached_median_std',
    #              '_cached_mean_std', '_cached_focal_point',
    #              '_cached_bilinear_std', '_cached_data')

    def __init__(self, data_complete, llx, lly, length):
        self.llx = int(llx)
        self.lly = int(lly)
        self.length = int(length)

        self._data_complete = data_complete

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
    def focal_point(self):
        return (self.llx + self.length/2, self.lly + self.length/2)

    @property_cached
    def bilinear_std(self):
        raise NotImplementedError('Bilinear fit not implemented')

    @property_cached
    def data(self):
        return self._data_complete[self.llx:self.llx+self.length,
                                   self.lly:self.lly+self.length]

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

    def iterLeafsEval(self, eval_func, epsilon):
        if eval_func(self) < epsilon or self.children is None:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafsEval(eval_func, epsilon):
                    yield q

    def _iterSplitNode(self):
        if self.length == 1:
            yield None
        for _nx, _ny in ((0, 0), (0, 1), (1, 0), (1, 1)):
            _q = QuadNode(self._data_complete,
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
               self.children, self._data_complete

    def __setstate__(self, state):
        self.llx, self.lly, self.length,\
            self.children, self._data_complete = state

    def __str__(self):
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


class Quadtree(Subject):
    def __init__(self, scene, epsilon=None):
        Subject.__init__(self)

        self._split_methods = {
            'mean_std': ['Std from mean', lambda node: node.mean_std],
            'median_std': ['Std from median', lambda node: node.median_std],
            'std': ['Std', lambda node: node.std],
        }
        self._norm_methods = {
            'mean': lambda node: node.mean,
            'median': lambda node: node.median,
        }

        self._scene = scene
        self._data = self._scene.displacement

        self._epsilon = None
        self._leafs = None

        self._log = logging.getLogger('Quadtree')

        self.splitMethodChanged = Subject()
        self.setSplitMethod('median_std')

    def setSplitMethod(self, split_method, parallel=True):
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
        return num.mean([self._split_func(b) for b in self._base_nodes])

    @property_cached
    def _epsilon_limit(self):
        return self._epsilon_init * .2

    @epsilon.setter
    def epsilon(self, value):
        if value < self._epsilon_limit:
            self._log.info('Epsilon is out of bounds [%0.3f], epsilon_limits' %
                           (value, self._epsilon_limit))
            return
        self.leafs = None
        self._epsilon = value
        self._notify()
        return

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
            leafs.extend([l for l in b.iterLeafsEval(self._split_func,
                                                     self.epsilon)])
        self._log.info('Gathering leafs (%d) for epsilon %.3f [%0.8f s]' %
                       (len(leafs), self.epsilon, time.time()-t0))
        return leafs

    @property
    def leaf_means(self):
        return num.array([l.mean for l in self.leafs])

    @property
    def leaf_medians(self):
        return num.array([l.median for l in self.leafs])

    @property
    def leaf_focal_points(self):
        return num.array([l.focal_point for l in self.leafs])

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

        for ix in range(int(nx)):
            for iy in range(int(ny)):
                _cx = ix * init_length
                _cy = iy * init_length
                self._base_nodes.append(QuadNode(self._data,
                                                 _cx, _cy,
                                                 int(init_length)))

        if len(self._base_nodes) == 0:
            raise AssertionError('Could not init base nodes.')
        return self._base_nodes

    @property_cached
    def plot(self):
        from kite.plot2d import PlotQuadTree2D
        return PlotQuadTree2D(self)

    def getStaticTarget(self):
        raise NotImplementedError

    def dump(self):
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

__all__ = '''
Quadtree
'''.split()


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss(2000, 2000)

    for e in num.linspace(0.1, .00005, num=30):
        sc.quadtree.epsilon = e
    # qp = Plot2DQuadTree(qt, cmap='spectral')
    # qp.plot()
