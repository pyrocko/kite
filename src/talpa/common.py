from PySide import QtGui
from pyqtgraph import dockarea
import pyqtgraph as pg

import numpy as num


class ModelLayout(pg.GraphicsLayoutWidget):
    def __init__(self, model, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)
        self.model = model

        self.plots = [
            ModelPlot(
                model,
                title='North',
                component=lambda m: m.north),
            ModelPlot(
                model,
                title='East',
                component=lambda m: m.east),
            ModelPlot(
                model,
                title='Down',
                component=lambda m: m.down),
            ModelPlot(
                model,
                title='LOS',
                component=lambda m: m.displacement)]

        for ip, plt in enumerate(self.plots):
            row = ip / 2
            col = ip % 2 + 1

            self.addItem(plt, row=row, col=col)
            plt.showGrid(x=True, y=True)
            plt.hideAxis('bottom')
            plt.hideAxis('left')
            plt.vb.border = pg.mkPen(50, 50, 50)
            if ip != 0:
                plt.setXLink(self.plots[0])
                plt.setYLink(self.plots[0])

        def getAxis(plt, orientation, label):
            axis = pg.AxisItem(
                orientation=orientation,
                linkView=plt.vb)
            axis.setLabel(label, units='m')
            return axis

        plts = self.plots
        self.addItem(getAxis(plts[0], 'left', 'Northing'), row=0, col=0)
        self.addItem(getAxis(plts[1], 'left', 'Northing'), row=1, col=0)
        self.addItem(getAxis(plts[0], 'bottom', 'Easting'), row=2, col=1)
        self.addItem(getAxis(plts[1], 'bottom', 'Easting'), row=2, col=2)


class ModelPlot(pg.PlotItem):

    position = 'left'

    def __init__(self, model, component, title='Untitled'):
        pg.PlotItem.__init__(self)
        self.title = title
        self.model = model
        self.component = component

        self.setAspectLocked(True)
        self.setLabels(
            bottom=('Easting', 'm'),
            left=('Northing', 'm'))

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


class ModelDockarea(dockarea.DockArea):

    def __init__(self, model, *args, **kwargs):
        dockarea.DockArea.__init__(self)

        layout = ModelLayout(model)
        cmap = ColormapPlots(plot=layout.plots[0])
        for plt in layout.plots:
            cmap.addPlot(plt)

        cmap_dock = dockarea.Dock(
            'Colormap',
            widget=cmap)
        cmap_dock.setStretch(1, None)

        layout_dock = dockarea.Dock(
            'Model Sandbox',
            widget=layout)
        self.addDock(layout_dock, position='right')
        self.addDock(cmap_dock, position='right')


class ColormapPlots(pg.HistogramLUTWidget):

    position = 'right'

    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)
        self.plots = [plot]

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

        lvl_min = lvl_max = 0
        for plot in self.plots:
            plt_min = num.nanmin(plot.data)
            plt_max = num.nanmax(plot.data)
            lvl_max = lvl_max if plt_max < lvl_max else plt_max
            lvl_min = lvl_min if plt_min > lvl_min else plt_min

        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(cmap)
        self.setLevels(-abs_range, abs_range)

    def addPlot(self, plot):
        self.plots.append(plot)
        self.setSymColormap()
        image = plot.image

        # hist_pen = pg.mkPen((170, 57, 57, 255), width=1.)
        image.setLookupTable(self.getLookupTable)

        def updateLevels():
            image.setLevels(self.region.getRegion())

        self.sigLevelChangeFinished.connect(updateLevels)
        self.sigLevelsChanged.connect(updateLevels)
        updateLevels()


class SourcesList(QtGui.QDockWidget):

    def __init__(self, model, *args, **kwargs):
        QtGui.QDockWidget.__init__(self, *args, **kwargs)
        self.model = model

        self.list = QtGui.QListView(self)
        self.addWidget(self.list)
