from PySide import QtGui
from pyqtgraph import dockarea
import pyqtgraph as pg

import numpy as num


class ModelPerspective(pg.PlotWidget):

    position = 'left'

    def __init__(self, model, component, title='Untitled', position='bottom'):
        pg.PlotWidget.__init__(self)
        self.model = model
        self.position = position
        self.component = component
        self.title = title

        self.setAspectLocked(True)

        self.image = pg.ImageItem(
            None,
            autoDownsample=False,
            useOpenGL=True)
        self.addItem(self.image)
        self.update()

        self.transFromFrame()

    @property
    def data(self):
        return self.component(self.model)

    def update(self):
        self.image.updateImage(self.data.T)

    def transFromFrame(self):
        self.image.resetTransform()
        self.image.scale(
            self.model.frame.dE,
            self.model.frame.dN)

    def setGradientEditor(self, gradient_editor):
        ge = gradient_editor
        image = self.image

        hist_pen = pg.mkPen((170, 57, 57, 255), width=1.)
        image.setLookupTable(ge.getLookupTable)

        def updateLevels():
            image.setLevels(ge.region.getRegion())

        ge.sigLevelChangeFinished.connect(updateLevels)
        ge.sigLevelsChanged.connect(updateLevels)
        updateLevels()

        # def updateHistogram():
        #     h = image.getHistogram()
        #     if h[0] is None:
        #         return
        #     ge.hist_syn.setData(*h)

        # ge.hist_syn = pg.PlotDataItem(pen=hist_pen)
        # ge.hist_syn.rotate(90.)
        # ge.vb.addItem(ge.hist_syn)
        # updateHistogram()

        # image.sigImageChanged.connect(updateHistogram)


class ModelPlots(dockarea.DockArea):

    def __init__(self, model, *args, **kwargs):
        dockarea.DockArea.__init__(self)

        plt_docks = []
        north_plot = ModelPerspective(
            model,
            title='North',
            component=lambda m: m.north)
        east_plot = ModelPerspective(
            model,
            title='East',
            component=lambda m: m.east)
        down_plot = ModelPerspective(
            model,
            title='Down',
            component=lambda m: m.down)
        los_plot = ModelPerspective(
            model,
            title='LOS',
            component=lambda m: m.displacement)
        plots = [north_plot, east_plot, down_plot, los_plot]

        cmap = ColormapPlots(plot=los_plot)
        rels = ['right', 'bottom', 'right', 'top']
        for ip, plot in enumerate(plots):
            plot.setGradientEditor(cmap)

            if plot is not north_plot:
                plot.setXLink(north_plot)
                plot.setYLink(north_plot)

            plt_docks.append(dockarea.Dock(
                plot.title,
                widget=plot))

            rel_to = None if len(plt_docks) == 1 else plt_docks[-2]
            print rel_to, rels[ip]
            self.addDock(
                plt_docks[-1],
                position=rels[ip],
                realtiveTo=rel_to)

        cmap_dock = dockarea.Dock(
            'Colormap',
            widget=cmap)
        cmap_dock.setStretch(1, None)
        self.addDock(cmap_dock, position='right')


class ColormapPlots(pg.HistogramLUTWidget):

    position = 'right'

    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)
        self._plot = plot

        self.axis.setLabel('Displacement / m')

        zero_marker = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen='w',
            movable=False)
        zero_marker.setValue(0.)
        zero_marker.setZValue(1000)
        self.vb.addItem(zero_marker)

        self.axis.setLabel('Displacement / m')
        self.setSymColormap()

    def setSymColormap(self):
        cmap = {'ticks':
                [[0., (0, 0, 0, 255)],
                 [1e-3, (106, 0, 31, 255)],
                 [.5, (255, 255, 255, 255)],
                 [1., (8, 54, 104, 255)]],
                'mode': 'rgb'}
        cmap = {'ticks':
                [[0., (0, 0, 0)],
                 [1e-3, (172, 56, 56)],
                 [.5, (255, 255, 255)],
                 [1., (51, 53, 120)]],
                'mode': 'rgb'}
        lvl_min = num.nanmin(self._plot.data)
        lvl_max = num.nanmax(self._plot.data)
        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(cmap)
        self.setLevels(-abs_range, abs_range)


class SourcesList(QtGui.QDockWidget):

    def __init__(self, model, *args, **kwargs):
        QtGui.QDockWidget.__init__(self, *args, **kwargs)
        self.model = model

        self.list = QtGui.QListView(self)
        self.addWidget(self.list)
