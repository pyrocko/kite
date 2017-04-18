from PySide import QtGui, QtCore
from pyqtgraph import dockarea
import pyqtgraph as pg

from .config import getConfig
from .common import SourceROI

import numpy as num


config = getConfig()


class ModelSceneLayout(pg.GraphicsLayoutWidget):
    def __init__(self, sandbox, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)
        self.sandbox = sandbox

        self.plots = [
            DisplacementPlot(
                sandbox,
                title='North',
                component=lambda m: m.north),
            DisplacementPlot(
                sandbox,
                title='East',
                component=lambda m: m.east),
            DisplacementPlot(
                sandbox,
                title='Down',
                component=lambda m: m.down),
            DisplacementPlot(
                sandbox,
                title='LOS',
                component=lambda m: m.los_displacement)]
        self.plots[-1].addHintText()

        self._mov_sig = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=60, slot=self.mouseMoved)

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

        for plt in self.plots:
            plt.vb.menu = QtGui.QMenu(self)

    def resizeEvent(self, ev):
        pg.GraphicsLayoutWidget.resizeEvent(self, ev)
        if hasattr(self, 'plots'):
            viewbox = self.plots[0].getViewBox()
            viewbox.autoRange()

    @QtCore.Slot(object)
    def mouseMoved(self, event):
        self.sandbox.cursor_tracker.sigMouseMoved.emit(event)


class ModelReferenceLayout(pg.GraphicsLayoutWidget):
    def __init__(self, sandbox, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)
        self.sandbox = sandbox

        self.plots = [
            DisplacementPlot(
                sandbox,
                title='Scene Displacement',
                component=lambda m: m.reference.scene.displacement),
            DisplacementPlot(
                sandbox,
                title='Model Residual',
                component=lambda m: m.reference.difference)]
        self.plots[-1].addHintText()

        self._mov_sig = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=60, slot=self.mouseMoved)

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

    @QtCore.Slot(object)
    def mouseMoved(self, event):
        self.sandbox.cursor_tracker.sigMouseMoved.emit(event)


class CursorRect(QtGui.QGraphicsRectItem):
    pen = pg.mkPen((0, 0, 0, 120), width=1.)
    cursor = QtCore.QRectF(
        (QtCore.QPointF(-1.5, -1.5)),
        (QtCore.QPointF(1.5, 1.5)))

    def __init__(self):
        QtGui.QGraphicsRectItem.__init__(self, self.cursor)
        self.setPen(self.pen)
        self.setZValue(1e9)
        self.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations)


class DisplacementPlot(pg.PlotItem):

    def __init__(self, sandbox, component, title='Untitled'):
        pg.PlotItem.__init__(self)
        self.title = title
        self.sandbox = sandbox
        self.component = component

        self.cursor = CursorRect()
        self.addCursor()

        self.setAspectLocked(True)
        self.setLabels(
            bottom=('Easting', 'm'),
            left=('Northing', 'm'))

        self.image = pg.ImageItem(
            None,
            autoDownsample=False,
            useOpenGL=True)
        self.addItem(self.image)

        self.title_label = pg.LabelItem(
            text='<span style="color: #9E9E9E;">'
                 '%s</span>' % self.title,
            justify='right', size='10pt',
            parent=self)
        self.title_label.anchor(
            itemPos=(0., 0.),
            parentPos=(.01, .01))
        self.title_label.setOpacity(.6)

        self.sandbox.sigModelUpdated.connect(
            self.update)
        self.sandbox.sources.modelAboutToBeReset.connect(
            self.removeSourceROIS)
        self.sandbox.sources.modelReset.connect(
            self.addSourceROIS)

        self.update()

        self.rois = []
        self.addSourceROIS()

    @property
    def data(self):
        return self.component(self.sandbox.model)

    @QtCore.Slot()
    def update(self):
        self.image.updateImage(self.data.T)
        self.transFromFrame()

    def transFromFrame(self):
        self.image.resetTransform()
        self.image.scale(
            self.sandbox.frame.dE,
            self.sandbox.frame.dN)

    @QtCore.Slot()
    def addSourceROIS(self):
        self.rois = []
        index = QtCore.QModelIndex()
        for isrc in xrange(self.sandbox.sources.rowCount(index)):
            index = self.sandbox.sources.index(isrc, 0, index)
            roi = index.data(SourceROI)
            self.rois.append(roi)
            self.addItem(roi)

    @QtCore.Slot()
    def removeSourceROIS(self):
        if self.rois:
            for roi in self.rois:
                self.removeItem(roi)
            self.update()

    def addCursor(self):
        if config.show_cursor:
            self.sandbox.cursor_tracker.sigCursorMoved.connect(self.drawCursor)
            self.sandbox.cursor_tracker.sigMouseMoved.connect(self.mouseMoved)
            self.addItem(self.cursor)

    @QtCore.Slot(object)
    def mouseMoved(self, event):
        if self.vb.sceneBoundingRect().contains(event[0]):
            map_pos = self.vb.mapSceneToView(event[0])

            img_pos = self.image.mapFromScene(event[0])
            pE, pN = img_pos.x(), img_pos.y()
            if pE < 0 or pN < 0:
                value = 0.
            elif (pE > self.image.image.shape[0] or
                  pN > self.image.image.shape[1]):
                value = 0.
            else:
                value = self.image.image[int(img_pos.x()),
                                         int(img_pos.y())]
            self.sandbox.cursor_tracker.sigCursorMoved.emit((map_pos, value))
            self.cursor.hide()
        else:
            self.cursor.show()

    @QtCore.Slot(object)
    def drawCursor(self, pos):
        pos, _ = pos
        self.cursor.setPos(pos)

    def addHintText(self):
        self.hint_text = pg.LabelItem(
            text='',
            justify='right', size='8pt',
            parent=self)
        self.hint_text.anchor(
            itemPos=(1., 1.),
            parentPos=(1., 1.))
        self.hint_text.text_template =\
            '<span style="font-family: monospace; color: #fff;'\
            'background-color: #000;">'\
            'East {0:08.2f} m | North {1:08.2f} m | '\
            'Displacement {2:2.4f} m</span>'
        self.hint_text.setOpacity(.6)
        self.sandbox.cursor_tracker.sigCursorMoved.connect(self.updateHintText)

    @QtCore.Slot()
    def updateHintText(self, pos):
        pos, value = pos
        self.hint_text.setText(
            self.hint_text.text_template.format(pos.x(), pos.y(), value))


class ColormapPlots(pg.HistogramLUTWidget):

    def __init__(self):
        pg.HistogramLUTWidget.__init__(self, image=None)
        self.plots = []

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

    @QtCore.Slot()
    def setSymColormap(self):
        cmap = {'ticks':
                [[0, (106, 0, 31, 255)],
                 [.5, (255, 255, 255, 255)],
                 [1., (8, 54, 104, 255)]],
                'mode': 'rgb'}
        cmap = {'ticks':
                [[0, (172, 56, 56)],
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
        image = plot.image
        if not self.plots:
            self.setImageItem(image)

        self.plots.append(plot)
        self.setSymColormap()

        # hist_pen = pg.mkPen((170, 57, 57, 255), width=1.)
        image.setLookupTable(self.getLookupTable)

        def updateLevels():
            image.setLevels(self.region.getRegion())

        self.sigLevelChangeFinished.connect(updateLevels)
        self.sigLevelsChanged.connect(updateLevels)
        updateLevels()


class ModelSceneDockarea(dockarea.DockArea):

    def __init__(self, sandbox, *args, **kwargs):
        dockarea.DockArea.__init__(self)
        layout = ModelSceneLayout(sandbox)

        cmap = ColormapPlots()
        for plt in layout.plots:
            cmap.addPlot(plt)
        sandbox.sigModelUpdated.connect(
            cmap.setSymColormap)

        cmap_dock = dockarea.Dock(
            'Colormap',
            widget=cmap)
        cmap_dock.setStretch(1, None)

        layout_dock = dockarea.Dock(
            'Model Sandbox',
            widget=layout)
        self.addDock(layout_dock, position='right')
        self.addDock(cmap_dock, position='right')


class ModelReferenceDockarea(dockarea.DockArea):

    def __init__(self, sandbox, *args, **kwargs):
        dockarea.DockArea.__init__(self)
        layout = ModelReferenceLayout(sandbox)

        cmap = ColormapPlots()
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
