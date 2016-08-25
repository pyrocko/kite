import numpy as num
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import logging
import time

_DEFAULT_IMSHOW = {
    'cmap': 'RdBu',
    'aspect': 'equal'
}

_VALID_COMPONENTS = {
    'displacement': 'LOS Displacement',
    'theta': 'LOS Theta',
    'phi': 'LOS Phi',
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
    """Base class for matplotlib 2D plots
    """
    def __init__(self, scene, **kwargs):
        self._scene = scene
        self._data = None

        self.fig = None
        self.ax = None
        self._show_fig = False

        self.title = 'unnamed'

        self.setCanvas(**kwargs)

        self._log = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def setCanvas(self, **kwargs):
        """Set canvas to plot in

        :param figure: Matplotlib figure to plot in
        :type figure: :py:class:`matplotlib.Figure`
        :param axes: Matplotlib axes to plot in
        :type axes: :py:class:`matplotlib.Axes`
        :raises: TypeError
        """
        axes = kwargs.get('axes', None)
        figure = kwargs.get('figure', None)

        if self.fig is not None:
            return
        elif axes is None and figure is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self._show_fig = True
        elif isinstance(axes, plt.Axes):
            self.fig, self.ax = axes.get_figure(), axes
        elif isinstance(figure, plt.Figure):
            self.fig, self.ax = figure, figure.gca()
        else:
            raise TypeError('axes has to be of type matplotlib.Axes\n'
                            'figure has to be of type matplotlib.Figure')
        self.image = AxesImage(self.ax)
        self.ax.add_artist(self.image)

        return

    @property
    def data(self):
        """ Data passed to matplotlib.image.AxesImage """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.image.set_data(self.data)

    @data.getter
    def data(self):
        if self._data is None:
            return num.empty((50, 50))
        return self._data

    def _initImagePlot(self, **kwargs):
        """ Initiate the plot

        :param figure: Matplotlib figure to plot in
        :type figure: :py:class:`matplotlib.Figure`
        :param axes: Matplotlib axes to plot in
        :type axes: :py:class:`matplotlib.Axes`
        """
        self.setCanvas(**kwargs)

        self.setColormap(kwargs.get('cmap', 'RdBu'))

        self.ax.set_xlim((0, self._scene.utm_x.size))
        self.ax.set_ylim((0, self._scene.utm_y.size))
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        self.ax.set_title(self.title)

        def close_figure(ev):
            self.fig = None
            self.ax = None

        self.fig.canvas.mpl_connect('close_event', close_figure)

    def plot(self, **kwargs):
        """Placeholder in prototype class

        :param figure: Matplotlib figure to plot in
        :type figure: :py:class:`matplotlib.Figure`
        :param axes: Matplotlib axes to plot in
        :type axes: :py:class:`matplotlib.Axes`
        :param **kwargs: kwargs are passed into `plt.imshow`
        :type **kwargs: dict
        :raises: NotImplemented
        """
        raise NotImplemented
        self._initImagePlot(**kwargs)
        if self._show_fig:
            plt.show()

    def _updateImage(self):
        self.image.set_data(self.data)

    def setColormap(self, cmap='RdBu'):
        """Set matplotlib colormap

        :param cmap: matplotlib colormap name, defaults to 'RdBu'
        :type cmap: str, optional
        """
        self.image.set_cmap(cmap)
        self._updateImage()

    def setColormapAuto(self, symmetric=True):
        """Set colormap limits automatically

        :param symmetric: symmetric colormap around 0, defaults to True
        :type symmetric: bool, optional
        """
        if symmetric:
            vmax = num.nanmax(num.abs(self.data))
            vmin = -vmax
        else:
            vmax = num.nanmax(num.abs(self.data))
            vmin = num.nanmin(num.abs(self.data))
        self.setColormapLimits(vmin, vmax)

    def setColormapLimits(self, vmin=None, vmax=None):
        """Set colormap limits

        :param vmin: lower limit, defaults to None
        :type vmin: float, optional
        :param vmax: upper limit, defaults to None
        :type vmax: float, optional
        """
        self.image.set_clim(vmin, vmax)


class PlotDisplacement2D(Plot2D):
    """Plotting 2D displacements though Matplotlib
    """
    def __init__(self, scene, **kwargs):
        Plot2D.__init__(self, scene, **kwargs)

        self.components_available = {
            'displacement': 'LOS Displacement',
            'theta': 'LOS Theta',
            'phi': 'LOS Phi',
            'cartesian.dE': 'Displacement dE',
            'cartesian.dN': 'Displacement dN',
            'cartesian.dU': 'Displacement dU',
        }

    def plot(self, component='displacement', **kwargs):
        """Plots any component fom Scene
        The following components are recognizes

        - 'cartesian.dE'
        - 'cartesian.dN'
        - 'cartesian.dU'
        - 'displacement'
        - 'phi'
        - 'theta'

        :param **kwargs: Keyword args forwarded to `matplotlib.plt.imshow()`
        :type **kwargs: {dict}
        :param component: Component to plot
['cartesian.dE', 'cartesian.dN', 'cartesian.dU',
'displacement', 'phi', 'theta']
        :type component: {string}, optional
        :param axes: Axes instance to plot in, defaults to None
        :type axes: :py:class:`matplotlib.Axes`, optional
        :param figure: Figure instance to plot in, defaults to None
        :type figure: :py:class:`matplotlib.Figure`, optional
        :param **kwargs: kwargs are passed into `plt.imshow`
        :type **kwargs: dict
        :returns: Imshow instance
        :rtype: :py:class:`matplotlib.image.AxesImage`
        :raises: AttributeError
        """
        self.setComponent(component)
        self.title = self.components_available[component]
        self._initImagePlot(**kwargs)

        if self._show_fig:
            plt.show()

    def setComponent(self, component):
        """Set displacement component to plot

        :param component: Displacement component to plot in
['cartesian.dE', 'cartesian.dN', 'cartesian.dU',
'displacement', 'phi', 'theta']
        :type component: str
        :raises: AttributeError, AttributeError
        """
        try:
            if component not in self.components_available.keys():
                raise AttributeError('Invalid component %s' % component)
            self.data = eval('self._scene.%s' % component)
        except:
            raise AttributeError('Could not access component %s' % component)

    def availableComponents(self):
        return self.components_available


class PlotQuadTree2D(Plot2D):
    """Plotting 2D Quadtrees though Matplotlib
    """
    def __init__(self, quadtree, **kwargs):
        self._quadtree = quadtree

        Plot2D.__init__(self, quadtree._scene)

    def plot(self, **kwargs):
        """Plot current quadtree

        :param axes: Axes instance to plot in, defaults to None
        :type axes: [:py:class:`matplotlib.Axes`], optional
        :param figure: Figure instance to plot in, defaults to None
        :type figure: [:py:class:`matplotlib.Figure`], optional
        :param **kwargs: kwargs are passed into `plt.imshow`
        :type **kwargs: dict
        """
        self.data = self._quadtree.leaf_matrix_means
        self.title = 'Quadtree Means'

        self._initImagePlot(**kwargs)
        self._addInfoText()

        if self._show_fig:
            plt.show()

    def _addInfoText(self):
        """ Add number of leafs in self.ax """
        self.ax.text(.975, .975, '%d Leafs' % len(self._quadtree.leafs),
                     transform=self.ax.transAxes, ha='right', va='top')

    def interactive(self):
        """Simple interactive quadtree plot with matplot
        """
        from matplotlib.widgets import Slider

        def change_epsilon(e):
            self._quadtree.epsilon = e

        def close_figure(*args):
            self._quadtree.unsubscribe(self._update)

        def _update():
            t0 = time.time()

            self.ax.texts = []
            self._addInfoText()
            self.data = self._quadtree.leaf_matrix_means
            self.setColormapAuto()
            self.ax.draw_artist(self.image)

            self._log.info('Redrew %d leafs [%0.8f s]' %
                           (len(self._quadtree.leafs), time.time()-t0))

        self.ax.set_position([0.05, 0.15, 0.90, 0.8])
        ax_eps = self.fig.add_axes([0.05, 0.1, 0.90, 0.03])

        self.data = self._quadtree.leaf_matrix_means
        self.title = 'Quadtree Means - Interactive'

        self._initImagePlot()
        self._addInfoText()

        epsilon = Slider(ax_eps, 'Epsilon',
                         self._quadtree.epsilon - 1.*self._quadtree.epsilon,
                         self._quadtree.epsilon + 1.*self._quadtree.epsilon,
                         valinit=self._quadtree.epsilon, valfmt='%1.3f')

        # Catch events
        epsilon.on_changed(change_epsilon)
        self._quadtree.subscribe(_update)
        self.fig.canvas.mpl_connect('close_event', close_figure)

        plt.show()


__all__ = """
Plot2D
""".split()

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()
