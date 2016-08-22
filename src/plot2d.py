# import numpy as num
import numpy as num
import matplotlib.pyplot as plt
import matplotlib.patches
import logging
import time

_DEFAULT_IMSHOW = {
    'cmap': 'RdBu',
    'aspect': 'equal'
}

_VALID_COMPONENTS = {
    'displacement': 'Displacement LOS',
    'theta': 'Theta LOS',
    'phi': 'Phi LOS',
    'cartesian.dE': 'Displacement dE',
    'cartesian.dN': 'Displacement dN',
    'cartesian.dU': 'Displacement dU',
}


def _getAxes(axes):
    raise DeprecationWarning('To be removed!')
    if axes is None:
        return plt.subplots(1, 1)
    elif isinstance(axes, plt.Axes):
        return axes.get_figure(), axes
    else:
        raise TypeError('Axes has to be of type matplotlib.Axes')


def _finishPlot(figure=None, axes=None):
    """Show plot if figure nor axes is given """
    if isinstance(axes, plt.Axes) or isinstance(figure, plt.Figure):
        return None
    return plt.show()


def _setCanvas(obj, figure=None, axes=None):
    if axes is None and figure is None:
        obj.fig, obj.ax = plt.subplots(1, 1)
    elif isinstance(axes, plt.Axes):
        obj.fig, obj.ax = axes.get_figure(), axes
    elif isinstance(figure, plt.Figure):
        obj.fig, obj.ax = figure, figure.gca()
    else:
        raise TypeError('axes has to be of type matplotlib.Axes\n'
                        'figure has to be of type matplotlib.Figure')


class Plot2D(object):
    def __init__(self, scene):
        self._scene = scene
        self.title = 'Displacement'

        self.fig = None
        self.ax = None
        self._im = None
        self._cb = None

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def _decorateAxes(self):
        self.ax.set_title('%s\n%s' % (self.title,
                                      self._scene.meta.title))

    def _decorateImshow(self):
        array = self._im.get_array()
        _vmax = num.abs(array).max()

        self._im.set_clim(-_vmax, _vmax)
        self._im.set_extent(
                    (self._scene.utm_x.min(),
                     self._scene.utm_x.max(),
                     self._scene.utm_y.min(),
                     self._scene.utm_y.max()))

    def plot(self, component='displacement', axes=None, figure=None, **kwargs):
        """Plots any component fom Scene

        :param **kwargs: Keyword args forwarded to `matplotlib.plt.imshow()`
        :type **kwargs: {dict}
        :param component: Component to plot
            ['phi', 'cartesian.dU', 'displacement', 'cartesian.dE',
            'theta', 'cartesian.dN']`, defaults to `'displacement'`
        :type component: {string}, optional
        :param axes: Axes instance to plot in, defaults to None
        :type axes: [:py:class:`matplotlib.Axes`], optional
        :param figure: Figure instance to plot in, defaults to None
        :type figure: [:py:class:`matplotlib.Figure`], optional
        :returns: Imshow instance
        :rtype: {[:py:class:`matplotlib.image.AxesImage`]}
        :raises: AttributeError
        """
        try:
            if component not in _VALID_COMPONENTS.keys():
                raise AttributeError('Invalid component %s' % component)
            data = eval('self._scene.%s' % component)
        except:
            raise AttributeError('Could not access component %s' % component)

        _setCanvas(self, figure, axes)
        self._decorateAxes()

        self.colorbar_label = _VALID_COMPONENTS[component]

        _kwargs = _DEFAULT_IMSHOW.copy()
        _kwargs.update(kwargs)

        self._im = self.ax.imshow(data, **_kwargs)
        self._decorateImshow()

        self.ax.set_aspect('equal')

        if figure is None:
            self.addColorbar()

        _finishPlot(figure, axes)
        return self._im

    def addColorbar(self):
        self._cb = self.fig.colorbar(self._im)
        self._cb.set_label(self.colorbar_label)


class QuadLeafRectangle(matplotlib.patches.Rectangle):
    """Representation if Quadleaf matplotlib.patches.Rectangle

    Not used at the moment
    """
    __slots__ = ('_plotquadtree', 'leaf')

    def __init__(self, plotquadtree, leaf, **kwargs):
        matplotlib.patches.Rectangle.__init__(self, (0, 0), 0, 0, **kwargs)

        self._plotquadtree = plotquadtree
        self.leaf = leaf

        self.set_xy((self.leaf.llx, self.leaf.lly))
        self.set_height(self.leaf.length)
        self.set_width(self.leaf.length)

        # self.set_alpha(.5)
        self.set_color(self._plotquadtree.sm.to_rgba(self.leaf.median))
        if self._plotquadtree.ax is not None:
            self.set_transform(self._plotquadtree.ax.transData)


class Plot2DQuadTree(object):
    def __init__(self, quadtree, cmap='RdBu', **kwargs):
        from matplotlib import cm
        self._quadtree = quadtree
        self._rectangles = []

        self.fig = None
        self.ax = None

        self.sm = cm.ScalarMappable(cmap=cmap)
        self.sm.set_clim(-1, 1)

        self._log = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self, figure=None, axes=None, **kwargs):
        _setCanvas(self, figure, axes)
        _vmax = num.abs(self._quadtree.leaf_means).max()

        # self._updateRectangles()
        self._decorateAxes()
        self._addInfoText()
        self.ax.set_xlim((0, self._quadtree._scene.utm_x.size))
        self.ax.set_ylim((0, self._quadtree._scene.utm_y.size))
        self.im = self.ax.imshow(self._quadtree.leaf_matrix_means,
                                 cmap=self.sm.get_cmap())
        self.im.set_clim(-_vmax, _vmax)
        # self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        _finishPlot(figure, axes)

    def _addInfoText(self):
        self.ax.text(.975, .975, '%d Leafs' % len(self._quadtree.leafs),
                     transform=self.ax.transAxes, ha='right', va='top')

    def interactive(self):
        from matplotlib.widgets import Slider

        _setCanvas(self)

        def change_epsilon(e):
            self._quadtree.epsilon = e

        def close_figure(*args):
            self._quadtree.unsubscribe(self._update)

        self.ax.set_position([0.05, 0.15, 0.90, 0.8])
        ax_eps = self.fig.add_axes([0.05, 0.1, 0.90, 0.03])

        self.plot(axes=self.ax)

        epsilon = Slider(ax_eps, 'Epsilon',
                         self._quadtree.epsilon - 1.*self._quadtree.epsilon,
                         self._quadtree.epsilon + 1.*self._quadtree.epsilon,
                         valinit=self._quadtree.epsilon, valfmt='%1.3f')

        # Catch events
        epsilon.on_changed(change_epsilon)
        self._quadtree.subscribe(self._update)
        self.fig.canvas.mpl_connect('close_event', close_figure)

        plt.show()

    def _updateRectangles(self):
        for rect in self._rectangles:
            try:
                self.ax.artists.remove(rect)
            except ValueError:
                pass

        self._rectangles = [QuadLeafRectangle(self, leaf)
                            for leaf in self._quadtree.leafs]

        self.ax.artists.extend(self._rectangles)

    def _updateColormap(self):
        _vmax = num.abs(self._quadtree.leaf_means).max()
        self.sm.set_clim(-_vmax, _vmax)

    def _update(self):
        t0 = time.time()
        _vmax = num.abs(self._quadtree.leaf_means).max()

        # self._updateColormap()
        # self._updateRectangles()
        self.im.set_data(self._quadtree.leaf_matrix_means)
        self.im.set_clim(-_vmax, _vmax)

        self.ax.texts = []
        self._addInfoText()

        # Update figure.canvas
        self.ax.draw_artist(self.im)
        # self.fig.canvas.blit(self.ax.bbox)

        # self.collections = []
        # self.ax.scatter(*zip(*self._quadtree.focal_points), s=4, color='k')

        # self.ax.set_xlim((0, self._quadtree._scene.utm_x.size))
        # self.ax.set_ylim((0, self._quadtree._scene.utm_y.size))
        # self.ax.invert_yaxis()
        # self.ax.set_aspect('equal')

        self._log.info('Redrew %d leafs [%0.8f s]' %
                       (len(self._quadtree.leafs), time.time()-t0))

    def _decorateAxes(self):
        pass


__all__ = """
Plot2D
""".split()

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()
