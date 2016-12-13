import numpy as num
import time
from pyrocko import guts

from kite.meta import Subject, property_cached


class QuadNode(object):
    ''' A node (or *tile*) in hel by :class:`kite.Quadtree`.

    :param llr: Lower left corner row in :attr:`kite.Scene.displacement`
        matrix.
    :type llr: int
    :param llc: Lower left corner column in :attr:`kite.Scene.displacement`
        matrix.
    :type llc: int
    :param length: Length of node in from ``llr, llc`` in both dimensions
    :type length: int
    :param id: Unique id of node
    :type id: str
    :param children: Node's children
    :type children: List of :class:`kite.quadtree.QuadNode`
    '''
    def __init__(self, quadtree, llr, llc, length):
        self.llr = int(llr)
        self.llc = int(llc)
        self.length = int(length)

        self._quadtree = quadtree
        self._scene = self._quadtree._scene
        self._slice_rows = slice(self.llr, self.llr+self.length)
        self._slice_cols = slice(self.llc, self.llc+self.length)

        self.id = 'node_%d-%d_%d' % (self.llr, self.llc, self.length)
        self.children = None

    @property_cached
    def nan_fraction(self):
        ''' Fraction of NaN values within the tile
        :type: float
        '''
        return float(num.sum(num.isnan(self.displacement))) / \
            self.displacement.size

    @property_cached
    def mean(self):
        ''' Mean displacement
        :type: float
        '''
        return num.nanmean(self.displacement)

    @property_cached
    def median(self):
        ''' Median displacement
        :type: float
        '''
        return num.nanmedian(self.displacement)

    @property_cached
    def std(self):
        ''' Standard deviation of displacement
        :type: float
        '''
        return num.nanstd(self.displacement)

    @property_cached
    def var(self):
        ''' Variance of displacement
        :type: float
        '''
        return num.nanvar(self.displacement)

    @property_cached
    def median_std(self):
        ''' Standard deviation from median
        :type: float
        '''
        return num.nanstd(self.displacement - self.median)

    @property_cached
    def mean_std(self):
        ''' Standard deviation from mean
        :type: float
        '''
        return num.nanstd(self.displacement - self.mean)

    @property
    def weight(self):
        '''
        :getter: Absolute weight derived from :class:`kite.Covariance`
         - works on tree leafs only.
        :type: float
        '''
        return self._quadtree._scene.covariance.getLeafWeight(self)

    @property_cached
    def bilinear_std(self):
        ''' Bilinear standard deviation
        :type: float
        '''
        raise NotImplementedError('Bilinear fit not implemented')

    @property_cached
    def focal_point(self):
        ''' Node focal point in local coordinates
        :type: float - (east, north)
        '''
        E = num.median(self.gridE.compressed())
        N = num.median(self.gridN.compressed())
        return E, N

    @property_cached
    def displacement(self):
        ''' Displacement array from :attr:`kite.Scene.displacement`
        :type: :class:`numpy.array`
        '''
        return self._scene.displacement[self._slice_rows, self._slice_cols]

    @property_cached
    def displacement_masked(self):
        ''' Masked displacement,
            see :attr:`kite.quadtree.QuadNode.displacement`
        :type: :class:`numpy.array`
        '''
        return num.ma.masked_array(self.displacement,
                                   num.isnan(self.displacement),
                                   fill_value=num.nan)

    @property_cached
    def phi(self):
        ''' Median Phi angle, see :class:`kite.Scene`.
        :type: float
        '''
        phi = self._scene.phi[self._slice_rows, self._slice_cols]
        return num.median(phi[~num.isnan(self.displacement)])

    @property_cached
    def theta(self):
        ''' Median Theta angle, see :class:`kite.Scene`.
        :type: float
        '''
        theta = self._scene.theta[self._slice_rows, self._slice_cols]
        return num.median(theta[~num.isnan(self.displacement)])

    @property_cached
    def gridE(self):
        ''' Grid holding local east coordinates,
            see :attr:`kite.scene.Frame.gridE`.
        :type: :class:`numpy.array`
        '''
        return self._scene.frame.gridE[self._slice_rows, self._slice_cols]

    @property_cached
    def gridN(self):
        ''' Grid holding local north coordinates,
            see :attr:`kite.scene.Frame.gridN`.
        :type: :class:`numpy.array`
        '''
        return self._scene.frame.gridN[self._slice_rows, self._slice_cols]

    @property
    def llE(self):
        '''
        :getter: Lower left east coordinate in local coordinates (*meter*).
        :type: float
        '''
        return self._scene.frame.E[self.llc]

    @property
    def llN(self):
        '''
        :getter: Lower left north coordinate in local coordinates (*meter*).
        :type: float
        '''
        return self._scene.frame.N[self.llr]

    @property
    def sizeE(self):
        '''
        :getter: Size in eastern direction in *meters*.
        :type: float
        '''
        sizeE = self.length * self._scene.frame.dE
        if (self.llE + sizeE) > self._scene.frame.E.max():
            sizeE = self._scene.frame.E.max() - self.llE
        return sizeE

    @property
    def sizeN(self):
        '''
        :getter: Size in northern direction in *meters*.
        :type: float
        '''
        sizeN = self.length * self._scene.frame.dN
        if (self.llN + sizeN) > self._scene.frame.N.max():
            sizeN = self._scene.frame.N.max() - self.llN
        return sizeN

    def iterTree(self):
        ''' Iterator over the whole tree

        :yields: Children of it's own.
        :type: :class:`kite.quadtree.QuadNode`
        '''
        yield self
        if self.children is not None:
            for c in self.children:
                for rc in c.iterTree():
                    yield rc

    def iterLeafsEval(self):
        ''' Iterator over the leafs, evaluating parameters from
        :class:`kite.Quadtree` instance.

        :yields: Leafs fullfilling the tree's parameters.
        :type: :class:`kite.quadtree.QuadNode`
        '''
        if (self._quadtree._split_func(self) < self._quadtree.epsilon and
            not self.length > self._quadtree._tile_size_lim_px[1])\
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
        for _nr, _nc in ((0, 0), (0, 1), (1, 0), (1, 1)):
            n = QuadNode(self._quadtree,
                         self.llr + self.length/2 * _nr,
                         self.llc + self.length/2 * _nc,
                         self.length/2)
            if n.displacement.size == 0 or num.all(num.isnan(n.displacement)):
                n = None
                continue
            yield n

    def createTree(self, eval_func, epsilon_limit):
        ''' Create the tree from a set of basenodes, ignited by
        :class:`kite.Quadtree` instance.

        :param eval_func: Evaluation function passed
        :type eval_func: function
        :param epsilon_limit: Lower epsilon limit the tree is devided to
        :type epsilon_limit: float
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
        return self.llr, self.llc, self.length,\
               self.children, self._quadtree

    def __setstate__(self, state):
        self.llr, self.llc, self.length,\
            self.children, self._quadtree = state


class QuadtreeConfig(guts.Object):
    ''' Quadtree configuration object holding essential parameters used to
    reconstruct a particular tree
    '''
    split_method = guts.String.T(
        default='median_std',
        help='Tile split method, available methods '
             ' ``[\'mean_std\' \'median_std\' \'std\']``')
    epsilon = guts.Float.T(
        optional=True,
        help='Threshold for tile splitting, measure for '
             'quadtree nodes\' variance')
    nan_allowed = guts.Float.T(
        default=0.9,
        help='Allowed NaN fraction per tile')
    tile_size_min = guts.Float.T(
        default=250.,
        help='Minimum allowed tile size in *meter*')
    tile_size_max = guts.Float.T(
        default=25e3,
        help='Maximum allowed tile size in *meter*')


class Quadtree(object):
    """Quadtree for simplifying InSAR displacement data held in
    :class:`kite.scene.Scene`

    Post-earthquake InSAR displacement scenes can hold a vast amount of data,
    which is unsuiteable for use with modelling code. By simplifying the data
    systematicallc through a parametrized quadtree we can reduce the dataset to
    significant displacements and have high-resolution where it matters and
    lower resolution at regions with less or constant deformation.
    """
    evChanged = Subject()
    evConfigChanged = Subject()

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

        self._log = scene._log.getChild('Quadtree')
        self.setScene(scene)
        self.parseConfig(config)

        self._leafs = None
        self._scene.evConfigChanged.subscribe(self.parseConfig)

    def setScene(self, scene):
        self._scene = scene
        self._displacement = self._scene.displacement
        self.frame = self._scene.frame

    def parseConfig(self, config=None):
        if config is None:
            self.config = self._scene.config.quadtree
        else:
            self.config = config

        self.evChanged.mute()

        self.setSplitMethod(self.config.split_method)
        if self.config.epsilon is not None:
            self.epsilon = self.config.epsilon
        self.nan_allowed = self.config.nan_allowed
        self.tile_size_min = self.config.tile_size_min
        self.tile_size_max = self.config.tile_size_max

        self.evChanged.unmute()

        self.evConfigChanged.notify()
        self.evChanged.notify()

    def setSplitMethod(self, split_method):
        """Set splitting method for quadtree tiles

        * ``mean_std``: tiles standard deviation from tile's mean is evaluated
        * ``median_std``: tiles standard deviation from tile's median is
            evaluated
        * ``std``: tiles standard deviation is evaluated

        :param split_method: Choose from methods
            ``['mean_std', 'median_std', 'std']``
        :type split_method: str
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
        self._log.debug('Changed to split method %s' % split_method)

    def _initTree(self):
        t0 = time.time()
        for b in self._base_nodes:
            b.createTree(self._split_func, self.epsilon_limit)

        self._log.debug('Tree created, %d nodes [%0.8f s]' % (self.nnodes,
                                                              time.time()-t0))

    @property
    def epsilon(self):
        """ Threshold for quadtree splitting the :class:`QuadNode`
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

        self.evChanged.notify()
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
        if (value > 1. or value < 0.):
            self._log.warning('NaN fraction must be 0 <= nan_allowed <=1')
            return

        self.leafs = None
        self.config.nan_allowed = value
        self.evChanged.notify()

    @property
    def tile_size_min(self):
        """Minimum allowed tile size in *meter*.

        :getter: Returns the minimum allowed tile size
        :setter: Sets the threshold
        :type: float
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
        """Maximum allowed tile size in *meter*.

        :getter: Returns the maximum allowed tile size
        :setter: Sets the threshold
        :type: float
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
        self.evChanged.notify()

    @property_cached
    def _tile_size_lim_px(self):
        dpx = self._scene.frame.dE
        return (int(self.tile_size_min / dpx),
                int(self.tile_size_max / dpx))

    @property
    def nnodes(self):
        """ Number of nodes of the built tree.
        :type: int
        """
        nnodes = 0
        for b in self._base_nodes:
            for n in b.iterTree():
                nnodes += 1
        return nnodes

    @property_cached
    def leafs(self):
        """
        :getter: List of leafs for current configuration.
        :type: (list[:class:`kite.quadtree.QuadNode`])
        """
        t0 = time.time()
        leafs = []
        for b in self._base_nodes:
            leafs.extend([l for l in b.iterLeafsEval()
                          if l.nan_fraction < self.nan_allowed])
        self._log.debug('Gathering leafs (%d) for epsilon %.4f [%0.8f s]' %
                        (len(leafs), self.epsilon, time.time()-t0))
        return leafs

    @property
    def nleafs(self):
        """
        :getter: Number of leafs for current parametrisation.
        :type: int
        """
        return len(self.leafs)

    @property
    def leaf_means(self):
        """
        :getter: Leaf mean displacements from
            :attr:`kite.quadtree.QuadNode.mean`.
        :type: :class:`numpy.ndarray`, size ``N``.
        """
        return num.array([l.mean for l in self.leafs])

    @property
    def leaf_medians(self):
        """
        :getter: Leaf median displacements from
            :attr:`kite.quadtree.QuadNode.median`.
        :type: :class:`numpy.ndarray`, size ``N``.
        """
        return num.array([l.median for l in self.leafs])

    @property
    def _leaf_focal_points(self):
        return num.array([l._focal_point for l in self.leafs])

    @property
    def leaf_focal_points(self):
        """
        :getter: Leaf focal points in local coordinates.
        :type: :class:`numpy.ndarray`, size ``(2,N)``
        """
        return num.array([l.focal_point for l in self.leafs])

    @property
    def leaf_matrix_means(self):
        """
        :getter: Leaf mean displacements casted to
            :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N,M)``
        """
        return self._getLeafsNormMatrix(method='mean')

    @property
    def leaf_matrix_medians(self):
        """
        :getter: Leaf median displacements casted to
            :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N,M)``
        """
        return self._getLeafsNormMatrix(method='median')

    @property
    def leaf_matrix_weights(self):
        """
        :getter: Leaf weights casted to :attr:`kite.Scene.displacement`.
        :type: :class:`numpy.ndarray`, size ``(N,M)``
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
        self._log.debug('Creating %d base nodes' % (nx * ny))

        for ir in xrange(int(nx)):
            for ic in xrange(int(ny)):
                llr = ir * init_length
                llc = ic * init_length
                self._base_nodes.append(QuadNode(self, llr, llc, init_length))

        if len(self._base_nodes) == 0:
            raise AssertionError('Could not init base nodes.')
        return self._base_nodes

    @property_cached
    def plot(self):
        """Simple `matplotlib` illustration of the quadtree

        :type: :class:`Quadtree.leaf_matrix_means`.
        """
        from kite.plot2d import QuadtreePlot
        return QuadtreePlot(self)

    def getStaticTarget(self):
        """Not Implemented
        """
        raise NotImplementedError

    def export(self, filename):
        """Export quadtree leafs in a *CSV* format to ``filename``.

        The format is as::

            # node_id, focal_point_E, focal_point_N, theta, phi,
                mean_displacement, median_displacement, absolute_weight

        :param filename: export to path
        :type filename: string
        """
        self._log.debug('Exporting quadtree to %s' % filename)
        with open(filename, mode='w') as qt_export:
            qt_export.write(
                '# node_id, focal_point_E, focal_point_N, theta, phi, '
                'mean_displacement, median_displacement, absolute_weight\n')
            for l in self.leafs:
                qt_export.write(
                    '{l.id}, {l.focal_point[0]}, {l.focal_point[1]}, '
                    '{l.theta}, {l.phi}, '
                    '{l.mean}, {l.median}, {l.weight}\n'.format(l=l))


__all__ = ['Quadtree', 'QuadtreeConfig']


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss(2000, 2000)

    for e in num.linspace(0.1, .00005, num=30):
        sc.quadtree.epsilon = e
    # qp = Plot2DQuadTree(qt, cmap='spectral')
    # qp.plot()
