import numpy as num
import matplotlib.pyplot as plt
from kite.meta import Subject
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


class Plot2D(Subject):
    """Base class for matplotlib 2D plots
    """
    def __init__(self, scene, **kwargs):
        Subject.__init__(self)
        self._scene = scene
        self._data = None

        self.fig = None
        self.ax = None
        self._show_plt = False
        self._colormap_symmetric = True

        self.title = 'unnamed'

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

        if isinstance(axes, plt.Axes):
            self.fig, self.ax = axes.get_figure(), axes
            self._show_plt = False
        elif isinstance(figure, plt.Figure):
            self.fig, self.ax = figure, figure.gca()
            self._show_plt = False
        elif axes is None and figure is None and self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self._show_plt = True
        else:
            raise TypeError('axes has to be of type matplotlib.Axes. '
                            'figure has to be of type matplotlib.Figure')
        self.image = AxesImage(self.ax)
        self.ax.add_artist(self.image)

    @property
    def data(self):
        """ Data passed to matplotlib.image.AxesImage """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.image.set_data(self.data)
        self.colormapAdjust()

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
        self.colormapAdjust()

        self.ax.set_xlim((0, self._scene.utm_x.size))
        self.ax.set_ylim((0, self._scene.utm_y.size))
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        self.ax.set_title(self.title)

        def close_figure(ev):
            self.fig = None
            self.ax = None
        try:
            self.fig.canvas.mpl_connect('close_event', close_figure)
        except:
            pass

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
        if self._show_plt:
            plt.show()

    def _updateImage(self):
        self.image.set_data(self.data)

    def setColormap(self, cmap='RdBu'):
        """Set matplotlib colormap

        :param cmap: matplotlib colormap name, defaults to 'RdBu'
        :type cmap: str, optional
        """
        self.image.set_cmap(cmap)
        self._notify()

    def colormapAdjust(self):
        """Set colormap limits automatically

        :param symmetric: symmetric colormap around 0, defaults to True
        :type symmetric: bool, optional
        """
        vmax = num.nanmax(self.data)
        vmin = num.nanmin(self.data)
        self.colormap_limits = (vmin, vmax)

    @property
    def colormap_symmetric(self):
        return self._colormap_symmetric

    @colormap_symmetric.setter
    def colormap_symmetric(self, value):
        self._colormap_symmetric = value
        self.colormapAdjust()

    @property
    def colormap_limits(self):
        return self.image.get_clim()

    @colormap_limits.setter
    def colormap_limits(self, limits):
        if not isinstance(limits, tuple):
            raise AttributeError('Limits have to be a tuple (vmin, vmax)')
        vmin, vmax = limits

        if self.colormap_symmetric:
            _max = max(abs(vmin), abs(vmax))
            vmin, vmax = -_max, _max
        self.image.set_clim(vmin, vmax)

        self._notify()

    @staticmethod
    def _colormapsAvailable():
        return [  # ('Perceptually Uniform Sequential',
                #  ['viridis', 'inferno', 'plasma', 'magma']),
                # ('Sequential', ['Blues', 'BuGn', 'BuPu',
                #                 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                #                 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                #               'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
                # ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                #                     'copper', 'gist_heat', 'gray', 'hot',
                #                     'pink', 'spring', 'summer', 'winter']),
                ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn',
                               'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                               'seismic', 'PuOr']),
                ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
                                 'Pastel2', 'Set1', 'Set2', 'Set3']),
                # ('Miscellaneous', ['gist_earth', 'terrain', 'ocean',
                #                  'brg', 'CMRmap', 'cubehelix', 'gist_stern',
                #                    'gnuplot', 'gnuplot2', 'gist_ncar',
                #                    'nipy_spectral', 'jet', 'rainbow',
                #                    'gist_rainbow', 'hsv', 'flag', 'prism'])
                ]


class PlotDisplacement2D(Plot2D):
    """Plotting 2D displacements though Matplotlib
    """
    def __init__(self, scene, **kwargs):
        Plot2D.__init__(self, scene, **kwargs)

        self.components_available = {
            'displacement': {
                'name': 'LOS Displacement',
                'eval': lambda sc: sc.displacement,
                },
            'theta': {
                'name': 'LOS Theta',
                'eval': lambda sc: sc.theta,
                },
            'phi': {
                'name': 'LOS Phi',
                'eval': lambda sc: sc.phi,
                },
            'dE': {
                'name': 'Displacement dE',
                'eval': lambda sc: sc.cartesian.dE,
                },
            'dN': {
                'name': 'Displacement dN',
                'eval': lambda sc: sc.cartesian.dN
                },
            'dU': {
                'name': 'Displacement dU',
                'eval': lambda sc: sc.cartesian.dU,
                },
        }

        self._component = 'displacement'

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
        self._initImagePlot(**kwargs)
        self.component = component
        self.title = self.components_available[component]

        if self._show_plt:
            plt.show()

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        try:
            if component not in self.components_available.keys():
                raise AttributeError('Invalid component %s' % component)
            self.data = self.components_available[component]['eval'](
                                                                self._scene)
        except AttributeError:
            raise AttributeError('Could not access component %s' % component)


class PlotQuadTree2D(Plot2D):
    """Plotting 2D Quadtrees though Matplotlib
    """
    def __init__(self, quadtree, **kwargs):
        self._quadtree = quadtree

        self.components_available = {
            'mean': {
                'name': 'Mean',
                'eval': lambda qt: qt.leaf_matrix_means,
                },
            'median': {
                'name': 'Median',
                'eval': lambda qt: qt.leaf_matrix_medians,
                },
        }
        self._component = 'mean'

        Plot2D.__init__(self, quadtree._scene)

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        try:
            if component not in self.components_available.keys():
                raise AttributeError('Invalid component %s' % component)
            self.data = self.components_available[component]['eval'](
                                                                self._quadtree)
        except AttributeError:
            raise AttributeError('Could not access component %s' % component)

    def plot(self, **kwargs):
        """Plot current quadtree

        :param axes: Axes instance to plot in, defaults to None
        :type axes: [:py:class:`matplotlib.Axes`], optional
        :param figure: Figure instance to plot in, defaults to None
        :type figure: [:py:class:`matplotlib.Figure`], optional
        :param **kwargs: kwargs are passed into `plt.imshow`
        :type **kwargs: dict
        """
        self._initImagePlot(**kwargs)
        self.data = self._quadtree.leaf_matrix_means
        self.title = 'Quadtree Means'

        self._addInfoText()

        if self._show_plt:
            plt.show()

    def _addInfoText(self):
        """ Add number of leafs in self.ax """
        self.ax.text(.975, .975, '%d Leafs' % len(self._quadtree.leafs),
                     transform=self.ax.transAxes, ha='right', va='top')

    def _update(self):
            t0 = time.time()

            self.ax.texts = []
            self._addInfoText()
            self.data = self._quadtree.leaf_matrix_means
            self.colormapAdjust()
            self.ax.draw_artist(self.image)

            self._log.info('Redrew %d leafs [%0.8f s]' %
                           (len(self._quadtree.leafs), time.time()-t0))

    def interactive(self):
        """Simple interactive quadtree plot with matplot
        """
        from matplotlib.widgets import Slider
        self._initImagePlot()

        def change_epsilon(e):
            self._quadtree.epsilon = e

        def close_figure(*args):
            self._quadtree.unsubscribe(self._update)

        self.ax.set_position([0.05, 0.15, 0.90, 0.8])
        ax_eps = self.fig.add_axes([0.05, 0.1, 0.90, 0.03])

        self.data = self._quadtree.leaf_matrix_means
        self.title = 'Quadtree Means - Interactive'

        self._addInfoText()

        epsilon = Slider(ax_eps, 'Epsilon',
                         self._quadtree.epsilon - 1.*self._quadtree.epsilon,
                         self._quadtree.epsilon + 1.*self._quadtree.epsilon,
                         valinit=self._quadtree.epsilon, valfmt='%1.3f')

        # Catch events
        epsilon.on_changed(change_epsilon)
        self._quadtree.subscribe(self._update)
        self.fig.canvas.mpl_connect('close_event', close_figure)

        plt.show()


__all__ = """
Plot2D
""".split()

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()
