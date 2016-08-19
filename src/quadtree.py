import numpy as num
import logging
import time
# import importlib

from kite.meta import Subject, property_cached


class QuadNode(object):
    """A Node in the Quadtree
    """
    # __slots__ = ('parent', '_tree', '_children',
    #              'llx', 'lly', 'length',
    #              'data', '_mean', '_median', 'std', '_var')

    def __init__(self, tree, llx, lly, length, parent=None):

        self.parent = parent
        self._tree = tree
        self._children = None

        self.llx = int(llx)
        self.lly = int(lly)
        self.length = int(length)

        self.data = self._tree.data[self.llx:self.llx+self.length,
                                    self.lly:self.lly+self.length]

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

    @property
    def children(self):
        return self._children

    def iterLeafs(self):
        if self._children is None:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafs():
                    yield q

    def _iterSplitNode(self):
        if self.length == 1:
            yield None
        for _nx, _ny in ((0, 0), (0, 1), (1, 0), (1, 1)):
            _q = QuadNode(self._tree,
                          self.llx + self.length/2 * _nx,
                          self.lly + self.length/2 * _ny,
                          self.length/2, parent=self)
            if _q.data.size == 0 or num.isnan(_q.data).all():
                continue
            yield _q

    def evaluateNode(self, eval_func):
        if eval_func(self) > self._tree.epsilon:  # or\
            # self.length > .1 * max(self._tree.data.shape): !! Very Expensive
            if self._children is None:
                self._children = [c for c in self._iterSplitNode()]
            for c in self._children:
                c.evaluateNode(eval_func)
        else:
            self._children = None

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


class Quadtree(Subject):
    def __init__(self, scene, epsilon=None):
        Subject.__init__(self)

        self.methods = {
            'mean_std': lambda node: node.mean_std,
            'median_std': lambda node: node.median_std,
            'std': lambda node: node.std,
        }

        self._scene = scene
        self.data = self._scene.displacement
        self._full_data_std = num.nanstd(self.data)

        self._epsilon = None
        self._leafs = None

        self._log = logging.getLogger('Quadtree')

        self.setMethod('median_std')

    def setMethod(self, method):
        if method not in self.methods.keys():
            raise AttributeError('Method %s not in %s'
                                 % (method, self.methods))
        self.method = method
        self.epsilon = num.mean([self.methods[self.method](b)
                                 for b in self.base_nodes])

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value < 0.1 * self._full_data_std:
            self._log.info('Epsilon is out of bounds [%0.3f]' % value)
            return
        self._epsilon = value
        self._leafs = None
        self._log.info('Changing epsilon to %0.3f' % value)
        t0 = time.time()

        for b in self.base_nodes:
            b.evaluateNode(self.methods[self.method])

        self._log.info('New tree has %d leafs [%0.8f s]' %
                       (len(self.leafs), (time.time()-t0)))
        self._notify()
        return

    @property
    def leafs(self):
        if self._leafs is None:
            self._leafs = []
            for n in self.base_nodes:
                self._leafs.extend([c for c in n.iterLeafs()])
        return self._leafs

    @property
    def means(self):
        return num.array([n.mean for n in self.leafs])

    @property
    def medians(self):
        return num.array([n.median for n in self.leafs])

    @property
    def focal_points(self):
        return num.array([n.focal_point for n in self.leafs])

    @property_cached
    def base_nodes(self):
        self._base_nodes = []
        init_length = num.power(2, num.ceil(num.log(num.min(self.data.shape)) /
                                            num.log(2)))
        nx, ny = num.ceil(num.array(self.data.shape)/init_length)

        for ix in range(int(nx)):
            for iy in range(int(ny)):
                _cx = ix * init_length
                _cy = iy * init_length
                self._base_nodes.append(QuadNode(self, _cx, _cy,
                                        int(init_length)))

        if len(self._base_nodes) == 0:
            raise AssertionError('Could not init base nodes.')
        return self._base_nodes

    @property_cached
    def plot(self):
        from kite.plot2d import Plot2DQuadTree
        return Plot2DQuadTree(self)

    def __str__(self):
        return '''
Quadtree for %s
  Initiated: %s
  Epsilon: %0.3f
  nLeafs:  %d
        ''' % (repr(self._scene), (self._base_nodes is not None),
               self.epsilon, len(self.leafs))

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
