import numpy as num
import logging
import time
# import importlib

from kite.meta import Subject


class QuadNode(object):
    """A Node in the Quadtree
    """
    __slots__ = ('parent', '_tree', '_children',
                 'llx', 'lly', 'length',
                 'data', '_mean', '_median', 'std', '_var')

    def __init__(self, tree, llx, lly, length, parent=None):

        self.parent = parent
        self._tree = tree
        self._children = None

        self.llx = int(llx)
        self.lly = int(lly)
        self.length = int(length)

        self.data = self._tree.data[self.llx:self.llx+self.length,
                                    self.lly:self.lly+self.length]
        self.std = num.nanstd(self.data)
        # Caching slots
        self._mean = None
        self._median = None
        self._var = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = num.nanmean(self.data)
        return self._mean

    @property
    def median(self):
        if self._median is None:
            self._median = num.nanmedian(self.data)
        return self._median

    @property
    def var(self):
        if self._var is None:
            self._var = num.nanvar(self.data)
        return self._var

    @property
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

    def _split_iter(self):
        if self.length == 1:
            yield ()
        for _nx, _ny in ((0, 0), (0, 1), (1, 0), (1, 1)):
            _q = QuadNode(self._tree,
                          self.llx + self.length/2 * _nx,
                          self.lly + self.length/2 * _ny,
                          self.length/2, parent=self)
            if _q.data.size == 0 or num.isnan(_q.data).all():
                continue
            yield _q

    def evaluateNode(self):
        if self.std > self._tree.epsilon:
            if self._children is None:
                self._children = [c for c in self._split_iter()]
            for c in self._children:
                c.evaluateNode()
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

        # self.mp = importlib.import_module('multiprocessing')

        self._scene = scene

        self.data = self._scene.displacement

        self._base_nodes = None
        self._epsilon = None

        self._leafs = None
        self._means = None
        self._focal_points = None
        self._plot = None

        self._full_data_std = num.nanstd(self.data)
        self._log = logging.getLogger('Quadtree')

        self.epsilon = 1.*self._full_data_std

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value < 0.1 * self._full_data_std:
            return

        self._epsilon = value

        self._leafs = None
        self._log.info('Changing epsilon to %0.3f' % value)
        t0 = time.time()

        for b in self.base_nodes:
            # self.mp.Process(target=b.evaluateNode).start()
            b.evaluateNode()
        # time.sleep(.1)
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
        return self._means

    @means.getter
    def means(self):
        return num.array([n.mean for n in self.leafs])

    @property
    def focal_points(self):
        return self._focal_points

    @focal_points.getter
    def focal_points(self):
        return num.array([n.focal_point for n in self.leafs])

    @property
    def base_nodes(self):
        if self._base_nodes is not None:
            return self._base_nodes

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

    @property
    def plot(self):
        if self._plot is None:
            from kite.plot2d import Plot2DQuadTree
            self._plot = Plot2DQuadTree(self)
        return self._plot

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
