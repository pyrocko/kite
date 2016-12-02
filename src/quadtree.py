import numpy as num
import time
from pyrocko import guts

from kite.meta import Subject, property_cached


class QuadNode(object):
    '''
    A node (or *tile*) in the Quadtree.
    '''
    def __init__(self, quadtree, llx, lly, length):
        self.llx = int(llx)
        self.lly = int(lly)
        self.length = int(length)

        self._quadtree = quadtree
        self._scene = self._quadtree._scene
        self._slice_rows = slice(self.llx, self.llx+self.length)
        self._slice_cols = slice(self.lly, self.lly+self.length)

        self.id = 'node_%d-%d_%d' % (self.llx, self.lly, self.length)
        self.children = None

    @property_cached
    def nan_fraction(self):
        ''' Fraction of NaN values within the tile '''
        return float(num.sum(num.isnan(self.displacement))) / \
            self.displacement.size

    @property_cached
    def mean(self):
        ''' Mean displacement '''
        return num.nanmean(self.displacement)

    @property_cached
    def median(self):
        ''' Median displacement '''
        return num.nanmedian(self.displacement)

    @property_cached
    def std(self):
        ''' Standard deviation of displacement '''
        return num.nanstd(self.displacement)

    @property_cached
    def var(self):
        ''' Variance of displacement '''
        return num.nanvar(self.displacement)

    @property_cached
    def median_std(self):
        ''' Standard deviation from median '''
        return num.nanstd(self.displacement - self.median)

    @property_cached
    def mean_std(self):
        ''' Standard deviation from mean '''
        return num.nanstd(self.displacement - self.mean)

    @property
    def weight(self):
        ''' Absolute weight derived from :py:class:`kite.covariance`
         - works on leaf only
    '''
        return self._quadtree._scene.covariance.getLeafWeight(self)

    @property_cached
    def bilinear_std(self):
        ''' Bilinear standard deviation '''
        raise NotImplementedError('Bilinear fit not implemented')

    @property_cached
    def focal_point(self):
        ''' Node focal point in local coordinates '''
        E = num.median(self.gridE.compressed())
        N = num.median(self.gridN.compressed())
        return E, N

    @property_cached
    def displacement(self):
        ''' Displacement array from :py:attr:`kite.Scene.displacement` '''
        return self._scene.displacement[self._slice_rows, self._slice_cols]

    @property_cached
    def displacement_masked(self):
        ''' Masked dispalcement,
        see :py:attr:`kite.quadtree.QuadNode.displacement`
        '''
        d = self._scene.displacement[self._slice_rows, self._slice_cols]
        return num.ma.masked_array(d, num.isnan(d), fill_value=num.nan)

    @property_cached
    def gridE(self):
        ''' Grid holding local East coordinates,
        see :py:attr:`kite.scene.Frame.gridE`
        '''
        return self._scene.frame.gridE[self._slice_rows, self._slice_cols]

    @property_cached
    def gridN(self):
        ''' Grid holding local North coordinates,
        see :py:attr:`kite.scene.Frame.gridN`
        '''
        return self._scene.frame.gridN[self._slice_rows, self._slice_cols]

    @property
    def llE(self):
        ''' Lower left East coordinate in local coordinates '''
        return self._scene.frame.E[self.lly]

    @property
    def llN(self):
        ''' Lower left East coordinate in local coordinates '''
        return self._scene.frame.N[self.llx]

    @property
    def sizeE(self):
        ''' Size in Eastern direction in meters '''
        sizeE = self.length * self._scene.frame.dE
        if (self.llE + sizeE) > self._scene.frame.E.max():
            sizeE = self._scene.frame.E.max() - self.llE
        return sizeE

    @property
    def sizeN(self):
        ''' Size in Northern direction in meters '''
        sizeN = self.length * self._scene.frame.dN
        if (self.llN + sizeN) > self._scene.frame.N.max():
            sizeN = self._scene.frame.N.max() - self.llN
        return sizeN

    def iterTree(self):
        ''' Iterator over the whole tree '''
        yield self
        if self.children is not None:
            for c in self.children:
                for rc in c.iterTree():
                    yield rc

    def iterLeafs(self):
        ''' CHECK! Iterator over the leafs - parameters from
        :py:class:`kite.quadtree.Quadtree` instance
        '''
        if self.children is None:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafs():
                    yield q

    def iterLeafsEval(self):
        ''' Iterator over the leafs - parameters from
        :py:class:`kite.quadtree.Quadtree` instance.
        '''
        if (self._quadtree._split_func(self) < self._quadtree.epsilon and
            (self._quadtree._tile_size_lim_px[1] is None or
             not self.length > self._quadtree._tile_size_lim_px[1]))\
           or self.children is None:
            yield self
        elif self.children[0].length < self._quadtree._tile_size_lim_px[0]:
            yield self
        else:
            for c in self.children:
                for q in c.iterLeafsEval():
                    yield q

    def _iterSplitNode(self):
        if self.length == 1:
            yield None
        for _nx, _ny in ((0, 0), (0, 1), (1, 0), (1, 1)):
            n = QuadNode(self._quadtree,
                         self.llx + self.length/2 * _nx,
                         self.lly + self.length/2 * _ny,
                         self.length/2)
            if n.displacement.size == 0 or num.all(num.isnan(n.displacement)):
                n = None
                continue
            yield n

    def createTree(self, eval_func, epsilon_limit):
        ''' Create the tree from a set of basenodes, ignited by
        :py:class:`kite.quadtree.Quadtree` instance.
        '''
        if (eval_func(self) > epsilon_limit or self.length >= 64)\
           and not self.length < 16:
            # self.length > .1 * max(self._quadtree._data.shape): !! Expensive
            self.children = [c for c in self._iterSplitNode()]
            for c in self.children:
                c.createTree(eval_func, epsilon_limit)
        else:
            self.children = None

    def __getstate__(self):
        return self.llx, self.lly, self.length,\
               self.children, self._quadtree

    def __setstate__(self, state):
        self.llx, self.lly, self.length,\
            self.children, self._quadtree = state


class QuadtreeConfig(guts.Object):
    split_method = guts.String.T(
        default='median_std',
        help='Tile split method, available methods '
             ' [\'mean_std\' \'median_std\' \'std\']')
    epsilon = guts.Float.T(
        default=-9999.,
        help='Threshold for tile splitting, measure for '
             'quadtree nodes\' variance')
    nan_allowed = guts.Float.T(
        default=0.9,
        help='Allowed NaN fraction per tile')
    tile_size_min = guts.Float.T(
        default=250.,
        help='Minimum allowed tile size in meter')
    tile_size_max = guts.Float.T(
        default=25e3,
        help='Maximum allowed tile size in meter')


class Quadtree(object):
    """Quadtree for simplifying InSAR displacement data held in
    :py:class:`kite.scene.Scene`

    Post-earthquake InSAR displacement scenes can hold a vast amount of data,
    which is unsuiteable for use with modelling code. By simplifying the data
    systematically through a parametrized quadtree we can reduce the dataset to
    significant displacements and have high-resolution where it matters and
    lower resolution at regions with less or constant deformation.
    """
    def __init__(self, scene, config=QuadtreeConfig()):
        self._split_methods = {
            'mean_std': ['Std around mean', lambda node: node.mean_std],
            'median_std': ['Std around median', lambda node: node.median_std],
            'std': ['Standard deviation (std)', lambda node: node.std],
        }
        self._norm_methods = {
            'mean': lambda node: node.mean,
            'median': lambda node: node.median,
            'weight': lambda node: node.weight,
        }

        self.treeUpdate = Subject()
        self.splitMethodChanged = Subject()
        self._log = scene._log.getChild('Quadtree')
        self.setScene(scene)
        self.parseConfig(config)

        self._leafs = None

    def setScene(self, scene):
        self._scene = scene
        self._displacement = self._scene.displacement
        self.frame = self._scene.frame

    def parseConfig(self, config):
        self.config = config
        self.setSplitMethod(self.config.split_method)
        if not self.config.epsilon == -9999.:
            self.epsilon = self.config.epsilon
        self.nan_allowed = self.config.nan_allowed
        self.tile_size_min = self.config.tile_size_min
        self.tile_size_max = self.config.tile_size_max

    def setSplitMethod(self, split_method, parallel=False):
        """Set splitting method for quadtree tiles

        * ``mean_std`` tiles standard deviation from tile's mean is evaluated
        * ``median_std`` tiles standard deviation from tile's median is
        evaluated
        * ``std`` tiles standard deviation is evaluated

        :param split_method: Choose from methods
        ``['mean_std', 'median_std', 'std']``
        :type split_method: {str}
        :raises: AttributeError
        """
        if split_method not in self._split_methods.keys():
            raise AttributeError('Method %s not in %s'
                                 % (split_method, self._split_methods))

        self.config.split_method = split_method
        self._split_func = self._split_methods[split_method][1]

        # Clearing cached properties through None
        self._epsilon_init = None
        self.epsilon_limit = None
        self.epsilon = self._epsilon_init

        self._initTree()
        self._log.info('Changed to split method %s' % split_method)
        self.splitMethodChanged._notify()

    def _initTree(self):
        t0 = time.time()
        for b in self._base_nodes:
            b.createTree(self._split_func, self.epsilon_limit)

        self._log.debug('Tree created, %d nodes [%0.8f s]' % (self.nnodes,
                                                              time.time()-t0))
        self.treeUpdate._notify()

    @property
    def epsilon(self):
        """ Threshold for quadtree splitting its ``QuadNode``
        """
        return self.config.epsilon

    @epsilon.setter
    def epsilon(self, value):
        value = float(value)
        if self.config.epsilon == value:
            return
        if value < self.epsilon_limit:
            self._log.warning(
                'Epsilon is out of bounds [%0.3f], epsilon_limit %0.3f' %
                (value, self.epsilon_limit))
            return
        self.leafs = None
        self.config.epsilon = value

        self.treeUpdate._notify()
        return

    @property_cached
    def _epsilon_init(self):
        return num.nanstd(self._displacement)
        # return num.mean([self._split_func(b) for b in self._base_nodes])

    @property_cached
    def epsilon_limit(self):
        """ Lowest allowed epsilon limit """
        return self._epsilon_init * .2

    @property
    def nan_allowed(self):
        """Fraction of allowed ``NaN`` values allwed in quadtree leafs, if
        value is exceeded the leaf is kicked out.
        """
        return self.config.nan_allowed

    @nan_allowed.setter
    def nan_allowed(self, value):
        if (value > 1. or value < 0.) and value != -9999.:
            raise AttributeError('NaN fraction must be 0 <= nan_allowed <=1')
        if value == 1.:
            value = -9999.

        self.leafs = None
        self.config.nan_allowed = value
        self.treeUpdate._notify()

    @property
    def tile_size_min(self):
        """Minimum allowed tile size in ``meter`.
        """
        return self.config.tile_size_min

    @tile_size_min.setter
    def tile_size_min(self, value):
        if value > self.tile_size_max:
            self._log.warning('tile_size_min > tile_size_max is required')
            return
        self.config.tile_size_min = value
        self._tileSizeChanged()

    @property
    def tile_size_max(self):
        """Minimum allowed tile size in ``meter`.
        """
        return self.config.tile_size_max

    @tile_size_max.setter
    def tile_size_max(self, value):
        if value < self.tile_size_min:
            self._log.warning('tile_size_min > tile_size_max is required')
            return
        self.config.tile_size_max = value
        self._tileSizeChanged()

    def _tileSizeChanged(self):
        self._tile_size_lim_px = None
        self.leafs = None
        self.treeUpdate._notify()

    @property_cached
    def _tile_size_lim_px(self):
        dpx = self._scene.frame.dE
        return (int(self.tile_size_min / dpx),
                int(self.tile_size_max / dpx))

    @property
    def nnodes(self):
        """Number of nodes in the quadtree instance.
        """
        nnodes = 0
        for b in self._base_nodes:
            for n in b.iterTree():
                nnodes += 1
        return nnodes

    @property_cached
    def leafs(self):
        """ Holds a list of current quadtrees leafs.
        """
        t0 = time.time()
        leafs = []
        for b in self._base_nodes:
            leafs.extend([l for l in b.iterLeafsEval()])
        if self.nan_allowed != -9999.:
            leafs[:] = [l for l in leafs if l.nan_fraction < self.nan_allowed]
        self._log.debug('Gathering leafs (%d) for epsilon %.4f [%0.8f s]' %
                        (len(leafs), self.epsilon, time.time()-t0))
        return leafs

    @property
    def nleafs(self):
        return len(self.leafs)

    @property
    def leaf_means(self):
        """Vector holding leafs mean displacement -
        :py:class:`numpy.ndarray`, size ``N``.
        """
        return num.array([l.mean for l in self.leafs])

    @property
    def leaf_medians(self):
        """Vector holding leafs median displacement -
        :py:class:`numpy.ndarray`, size ``N``.
        """
        return num.array([l.median for l in self.leafs])

    @property
    def _leaf_focal_points(self):
        return num.array([l._focal_point for l in self.leafs])

    @property
    def leaf_focal_points(self):
        """ Matrix holding leafs mean displacement -
        :py:class:`numpy.ndarray`, size ``(N, 2)``.
        """
        return num.array([l.focal_point for l in self.leafs])

    @property
    def leaf_matrix_means(self):
        """Matrix holding leafs mean values  -
        ``(N,M)`` like `Scene.displacement`.
        """
        return self._getLeafsNormMatrix(method='mean')

    @property
    def leaf_matrix_medians(self):
        """Matrix holding leafs median values -
        ``(N,M)`` like `Scene.displacement`.
        """
        return self._getLeafsNormMatrix(method='median')

    @property
    def leaf_matrix_weights(self):
        """Matrix holding leaf weights  -
        ``(N,M)`` like `Scene.displacement`.
        """
        return self._getLeafsNormMatrix(method='weight')

    def _getLeafsNormMatrix(self, method='median'):
        if method not in self._norm_methods.keys():
            raise AttributeError('Method %s is not in %s' % (method,
                                 self._norm_methods.keys()))

        leaf_matrix = num.empty_like(self._displacement)
        leaf_matrix.fill(num.nan)
        for l in self.leafs:
            leaf_matrix[l._slice_rows, l._slice_cols] = \
                self._norm_methods[method](l)
        leaf_matrix[num.isnan(self._displacement)] = num.nan
        return leaf_matrix

    @property_cached
    def _base_nodes(self):
        self._base_nodes = []
        init_length = num.power(
            2, num.ceil(num.log(num.min(self._displacement.shape))
                        / num.log(2)))
        nx, ny = num.ceil(num.array(self._displacement.shape)/init_length)

        for ix in xrange(int(nx)):
            for iy in xrange(int(ny)):
                llx = ix * init_length
                lly = iy * init_length
                self._base_nodes.append(QuadNode(self, llx, lly, init_length))

        if len(self._base_nodes) == 0:
            raise AssertionError('Could not init base nodes.')
        return self._base_nodes

    @property_cached
    def plot(self):
        """Simple `matplotlib` illustration of
        :py:class:`Quadtree.leaf_matrix_means`.
        """
        from kite.plot2d import QuadtreePlot
        return QuadtreePlot(self)

    def getStaticTarget(self):
        """Not Implemented
        """
        raise NotImplementedError


__all__ = ['Quadtree', 'QuadtreeConfig']


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss(2000, 2000)

    for e in num.linspace(0.1, .00005, num=30):
        sc.quadtree.epsilon = e
    # qp = Plot2DQuadTree(qt, cmap='spectral')
    # qp.plot()
