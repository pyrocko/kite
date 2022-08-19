import logging
import time

import matplotlib.pyplot as plt
import numpy as num
from matplotlib.image import AxesImage

from kite.util import Subject

__all__ = ["ScenePlot", "QuadtreePlot", "CovariancePlot"]

_DEFAULT_IMSHOW = {"cmap": "RdBu", "aspect": "equal"}

_VALID_COMPONENTS = {
    "displacement": "LOS Displacement",
    "theta": "LOS Theta",
    "phi": "LOS Phi",
    "cartesian.dE": "Displacement dE",
    "cartesian.dN": "Displacement dN",
    "cartesian.dU": "Displacement dU",
}


class Plot2D(object):
    """Base class for matplotlib 2D plots"""

    def __init__(self, scene, **kwargs):
        self.evPlotChanged = Subject()
        self._scene = scene
        self._data = None

        self.fig = None
        self.ax = None
        self._show_plt = False
        self._colormap_symmetric = True

        self.title = "unnamed"

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
        axes = kwargs.get("axes", None)
        figure = kwargs.get("figure", None)

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
            raise TypeError(
                "axes has to be of type matplotlib.Axes. "
                "figure has to be of type matplotlib.Figure"
            )
        self.image = AxesImage(self.ax)
        self.ax.add_artist(self.image)

    @property
    def data(self):
        """Data passed to matplotlib.image.AxesImage"""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.image.set_data(self.data)
        self.colormapAdjust()

    @data.getter
    def data(self):
        if self._data is None:
            return num.zeros((50, 50))
        return self._data

    def _initImagePlot(self, **kwargs):
        """Initiate the plot

        :param figure: Matplotlib figure to plot in
        :type figure: :py:class:`matplotlib.Figure`
        :param axes: Matplotlib axes to plot in
        :type axes: :py:class:`matplotlib.Axes`
        """
        self.setCanvas(**kwargs)

        self.setColormap(kwargs.get("cmap", "RdBu"))
        self.colormapAdjust()

        self.ax.set_xlim((0, self._scene.frame.E.size))
        self.ax.set_ylim((0, self._scene.frame.N.size))
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

        self.ax.set_title(self.title)

        def close_figure(ev):
            self.fig = None
            self.ax = None

        try:
            self.fig.canvas.mpl_connect("close_event", close_figure)
        except Exception:
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
        raise NotImplementedError()
        self._initImagePlot(**kwargs)
        if self._show_plt:
            plt.show()

    def _updateImage(self):
        self.image.set_data(self.data)

    def setColormap(self, cmap="RdBu"):
        """Set matplotlib colormap

        :param cmap: matplotlib colormap name, defaults to 'RdBu'
        :type cmap: str, optional
        """
        self.image.set_cmap(cmap)
        self.evPlotChanged.notify()

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
            raise AttributeError("Limits have to be a tuple (vmin, vmax)")
        vmin, vmax = limits

        if self.colormap_symmetric:
            _max = max(abs(vmin), abs(vmax))
            vmin, vmax = -_max, _max
        self.image.set_clim(vmin, vmax)

        self.evPlotChanged.notify()

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
            (
                "Diverging",
                [
                    "BrBG",
                    "bwr",
                    "coolwarm",
                    "PiYG",
                    "PRGn",
                    "RdBu",
                    "RdGy",
                    "RdYlBu",
                    "RdYlGn",
                    "Spectral",
                    "seismic",
                    "PuOr",
                ],
            ),
            (
                "Qualitative",
                [
                    "Accent",
                    "Dark2",
                    "Paired",
                    "Pastel1",
                    "Pastel2",
                    "Set1",
                    "Set2",
                    "Set3",
                ],
            ),
            # ('Miscellaneous', ['gist_earth', 'terrain', 'ocean',
            #                  'brg', 'CMRmap', 'cubehelix', 'gist_stern',
            #                    'gnuplot', 'gnuplot2', 'gist_ncar',
            #                    'nipy_spectral', 'jet', 'rainbow',
            #                    'gist_rainbow', 'hsv', 'flag', 'prism'])
        ]


class ScenePlot(Plot2D):
    """Plotting 2D displacements though Matplotlib"""

    def __init__(self, scene, **kwargs):
        Plot2D.__init__(self, scene, **kwargs)

        self.components_available = {
            "displacement": {
                "name": "LOS Displacement",
                "eval": lambda sc: sc.displacement,
            },
            "theta": {"name": "LOS Theta", "eval": lambda sc: sc.theta},
            "phi": {"name": "LOS Phi", "eval": lambda sc: sc.phi},
            "dE": {"name": "Displacement dE", "eval": lambda sc: sc.cartesian.dE},
            "dN": {"name": "Displacement dN", "eval": lambda sc: sc.cartesian.dN},
            "dU": {"name": "Displacement dU", "eval": lambda sc: sc.cartesian.dU},
        }

        self._component = "displacement"

    def plot(self, component="displacement", **kwargs):
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
                raise AttributeError("Invalid component %s" % component)
            self.data = self.components_available[component]["eval"](self._scene)
        except AttributeError:
            raise AttributeError("Could not access component %s" % component)


class QuadtreePlot(Plot2D):
    """Plotting 2D Quadtrees though Matplotlib"""

    def __init__(self, quadtree, **kwargs):
        self._quadtree = quadtree

        self.components_available = {
            "mean": {
                "name": "Mean",
                "eval": lambda qt: qt.leaf_matrix_means,
            },
            "median": {
                "name": "Median",
                "eval": lambda qt: qt.leaf_matrix_medians,
            },
        }
        self._component = "mean"

        Plot2D.__init__(self, quadtree.scene)

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        try:
            if component not in self.components_available.keys():
                raise AttributeError("Invalid component %s" % component)
            self.data = self.components_available[component]["eval"](self._quadtree)
        except AttributeError:
            raise AttributeError("Could not access component %s" % component)

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
        self.title = "Quadtree Means"

        self._addInfoText()

        if self._show_plt:
            plt.show()

    def _addInfoText(self):
        """Add number of leaves in self.ax"""
        self.ax.text(
            0.975,
            0.975,
            "%d Leafs" % len(self._quadtree.leaves),
            transform=self.ax.transAxes,
            ha="right",
            va="top",
        )

    def _update(self):
        t0 = time.time()

        self.ax.texts = []
        self._addInfoText()
        self.data = self._quadtree.leaf_matrix_means
        self.colormapAdjust()
        self.ax.draw_artist(self.image)

        self._log.info(
            "Redrew %d leaves [%0.8f s]"
            % (len(self._quadtree.leaves), time.time() - t0)
        )

    def interactive(self):
        """Simple interactive quadtree plot with matplot

        This is a relictic function
        """
        from matplotlib.widgets import Slider

        self._initImagePlot()

        def change_epsilon(e):
            self._quadtree.epsilon = e

        def close_figure(*args):
            self._quadtree.evChanged.unsubscribe(self._update)

        self.ax.set_position([0.05, 0.15, 0.90, 0.8])
        ax_eps = self.fig.add_axes([0.05, 0.1, 0.90, 0.03])

        self.data = self._quadtree.leaf_matrix_means
        self.title = "Quadtree Means - Interactive"

        self._addInfoText()

        epsilon = Slider(
            ax_eps,
            "Epsilon",
            self._quadtree.epsilon - 1.0 * self._quadtree.epsilon,
            self._quadtree.epsilon + 1.0 * self._quadtree.epsilon,
            valinit=self._quadtree.epsilon,
            valfmt="%1.3f",
        )

        # Catch events
        epsilon.on_changed(change_epsilon)
        self._quadtree.evChanged.subscribe(self._update)
        self.fig.canvas.mpl_connect("close_event", close_figure)

        plt.show()


class CovariancePlot(object):
    def __init__(self, covariance, *args, **kwargs):
        self._covariance = covariance
        self.variance = covariance.variance
        self.quadtree = covariance.quadtree
        self.scene = covariance.quadtree.scene
        self.fig = plt.figure(figsize=(11.692, 8.267))

        self.ax_noi = self.fig.add_subplot(221)
        self.ax_cov = self.fig.add_subplot(222)
        self.ax_svar = self.fig.add_subplot(224)
        self.ax_pow = self.fig.add_subplot(223)
        # self.ax_qud = self.fig.add_subplot(325)

        self.plotCovariance(self.ax_cov)
        self.plotSemivariogram(self.ax_svar)
        self.plotPowerspec(self.ax_pow)
        self.plotNoise(self.ax_noi)
        # self.plotQuadtreeWeight(self.ax_qud)

        # self.plotPowerfit()

        self.fig.subplots_adjust(
            left=0.05, bottom=0.075, right=0.95, top=0.95, wspace=0.2, hspace=0.25
        )

    def __call__(self):
        self.fig.show()

    def show(self):
        from matplotlib.pyplot import figure

        fig = figure(FigureClass=self)
        fig.show()

    def plotCovariance(self, ax):
        cov, d = self._covariance.getCovariance()
        var = num.empty_like(d)
        var.fill(self.variance)
        ax.plot(d, cov)

        # d_interp = num.linspace(d.min(), d.max()+10000., num=50)
        # ax.plot(d_interp, self._covariance.covariance(d_interp),
        #        label='Interpolated', ls='--')
        model = self._covariance.getModelFunction(d, *self._covariance.covariance_model)
        ax.plot(d, var, label="variance", ls="--")
        ax.plot(d, model, label="interpolated", ls="--")

        ax.legend(loc="best")
        ax.grid(alpha=0.4)
        ax.set_title("Covariogram")
        ax.set_xlabel("distance [$m$]")
        ax.set_ylabel("covariance [$m^2$]")

    def plotNoise(self, ax):
        noise_data = self._covariance.noise_data
        noise_coord = self._covariance.noise_coord

        ax.imshow(
            num.flipud(noise_data),
            aspect="equal",
            extent=(0, noise_coord[2], 0, noise_coord[3]),
        )

        ax.set_title("Noise Data")
        ax.set_xlabel("X [$m$]")
        ax.set_ylabel("Y [$m$]")

    def plotSemivariogram(self, ax):
        svar, d = self._covariance.structure_spatial
        var = num.empty_like(d)
        var.fill(self.variance)
        ax.plot(d, svar)
        ax.plot(d, var, label="variance", ls="--")

        ax.legend(loc=4)
        ax.grid(alpha=0.4)
        ax.set_title("Semi-Variogram")
        ax.set_xlabel("distance [$m$]")
        ax.set_ylabel("semi-variance [$m^2$]")

    def plotPowerspec(self, ax):
        (
            power_spec,
            k,
            dk,
            f_spec,
            k_x,
            k_y,
        ) = self._covariance.powerspecNoise1D()  # noiseSpectrum()

        # power_spec_x = num.mean(f_spec, axis=1)
        # power_spec_y = num.mean(f_spec, axis=0)

        # ax.plot(k_x[k_x > 0], power_spec_x[k_x > 0], label='$k_x$')
        # ax.plot(k_y[k_y > 0], power_spec_y[k_y > 0], label='$k_y$')
        ax.plot(k, power_spec, label="$k_{total}$")

        # ax.legend(loc=1)
        ax.grid(alpha=0.4, which="both")
        ax.set_title("Power Spectrum")
        ax.set_xlabel("wavenumber [$cycles/m$]")
        ax.set_xscale("log")
        ax.set_ylabel("power [$m^2$]")
        ax.set_yscale("log")

    def plotQuadtreeWeight(self, ax):
        # extent = (self._quadtree.frame.llE, self._quadtree.frame.urE,
        #           self._quadtree.frame.llN, self._quadtree.frame.urN)
        # cb = ax.imshow(self._quadtree.leaf_matrix_weights, aspect='equal',
        #                extent=extent)
        ax.set_xlabel("UTM X [$m$]")
        ax.set_ylabel("UTM Y [$m$]")

    def plotPowerfit(self):
        import scipy as sp

        def behaviour(k, a, b):
            return (k**a) / b

        def selectRegime(k, d1, d2):
            return num.logical_and(((1.0) / k) > d1, ((1.0) / k) < d2)

        def curve_fit(k, p_spec):
            return sp.optimize.curve_fit(
                behaviour,
                k,
                p_spec,
                p0=None,
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=(-num.inf, num.inf),
                method=None,
                jac=None,
            )

        def covar_analyt(p, k):
            ps = behaviour(k[k > 0], *p)
            cov = sp.fftpack.dct(ps, type=2, n=None, axis=-1, norm="ortho")
            d = num.arange(1, k[k > 0].size + 1) * self.scene.frame.dE
            return cov, d

        power_spec, k, f_spec, k_x, k_y = self._covariance.noiseSpectrum()
        # Regimes accord. to Hanssen, 2001
        reg1 = selectRegime(k_x, 2000.0, num.inf)
        reg2 = selectRegime(k_x, 500.0, 2000.0)
        reg3 = selectRegime(k_x, 10.0, 500.0)

        for i, r in enumerate([reg1, reg2, reg3]):
            p, cov = curve_fit(k[r], power_spec[r])
            self.ax_pow.plot(
                k[r],
                behaviour(k[r], *p),
                ls="--",
                lw=1.5,
                alpha=0.8,
                label="Fit regime %d" % (i + 1),
            )

            cov, d = covar_analyt(p, k)
            self.ax_cov.plot(
                d, cov, ls="--", lw=1.5, alpha=0.8, label="Fit regime %d" % (i + 1)
            )

        self.ax_cov.legend(loc=1)
        self.ax_pow.legend(loc=1)


class SyntheticNoisePlot(object):
    def __init__(self, covariance, *args, **kwargs):
        self._covariance = covariance
        self.noise_data = self._covariance.noise_data
        clim_max = num.nanmax(self.noise_data)
        clim_min = num.nanmin(self.noise_data)
        self.clim = min(abs(clim_max), abs(clim_min))
        self.fig = plt.figure(figsize=(6.692, 3.149))

        self.ax_noi = self.fig.add_subplot(121)
        self.ax_snoi = self.fig.add_subplot(122)

        self.plotSynthNoise(self.ax_snoi)
        self.plotNoise(self.ax_noi)
        # self.plotQuadtreeWeight(self.ax_qud)

        # self.plotPowerfit()

        self.fig.subplots_adjust(
            left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.25
        )

    def __call__(self):
        self.fig.show()

    def show(self):
        from matplotlib.pyplot import figure

        fig = figure(FigureClass=self)
        fig.show()

    def plotNoise(self, ax):
        noise_coord = self._covariance.noise_coord

        im = ax.imshow(
            num.flipud(self.noise_data),
            aspect="equal",
            extent=(0, noise_coord[2], 0, noise_coord[3]),
        )

        ax.set_title("Noise Data")
        ax.set_xlabel("X [$m$]")
        ax.set_ylabel("Y [$m$]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("LOS displacement [$m$]")
        im.set_clim(-self.clim, self.clim)

    def plotSynthNoise(self, ax):
        noise_shape = num.shape(self.noise_data)
        noise_data = self._covariance.syntheticNoise(noise_shape)
        noise_coord = self._covariance.noise_coord

        im = ax.imshow(
            num.flipud(noise_data),
            aspect="equal",
            extent=(0, noise_coord[2], 0, noise_coord[3]),
        )

        ax.set_title("Synthetic Noise")
        ax.set_xlabel("X [$m$]")
        ax.set_ylabel("Y [$m$]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("LOS displacement [$m$]")
        im.set_clim(-self.clim, self.clim)


if __name__ == "__main__":
    from kite.scene import SceneSynTest

    sc = SceneSynTest.createGauss()
