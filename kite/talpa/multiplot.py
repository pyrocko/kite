import numpy as num
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import dockarea

from .config import getConfig
from .util import SourceROI

d2r = num.pi / 180.0


class SandboxSceneLayout(pg.GraphicsLayoutWidget):

    PLOT_VIEWS = ["north", "east", "down", "los"]

    def __init__(self, sandbox, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)
        self.sandbox = sandbox

        self.plots = [
            DisplacementPlot(sandbox, title="North", component=lambda m: m.north),
            DisplacementPlot(sandbox, title="East", component=lambda m: m.east),
            DisplacementVectorPlot(sandbox, title="Down", component=lambda m: m.down),
            DisplacementPlot(sandbox, title="LOS", component=lambda m: m.displacement),
        ]

        for plt in self.plots:
            plt.vb.menu = QtWidgets.QMenu(self)

        self.updateViews()
        getConfig().qconfig.updated.connect(self.updateViews)

        self._mov_sig = pg.SignalProxy(
            self.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved
        )

    @QtCore.pyqtSlot()
    def updateViews(self):
        self.clear()
        for plt in self.plots:
            plt.removeHintText()

        config_mask = [
            getConfig().__getattribute__(cfg)
            for cfg in ("view_north", "view_east", "view_down", "view_los")
        ]

        visible_plots = [plt for ip, plt in enumerate(self.plots) if config_mask[ip]]

        for ip, plt in enumerate(visible_plots):
            row = ip / 2
            col = ip % 2 + 1

            self.addItem(plt, row=row, col=col)
            plt.showGrid(x=True, y=True)
            plt.hideAxis("bottom")
            plt.hideAxis("left")
            plt.vb.border = pg.mkPen(50, 50, 50)
            if ip != 0:
                plt.setXLink(self.plots[0])
                plt.setYLink(self.plots[0])

        if len(visible_plots) > 0:
            visible_plots[-1].addHintText()
            visible_plots[-1].autoRange(items=[visible_plots[-1].image])

    def resizeEvent(self, ev):
        pg.GraphicsLayoutWidget.resizeEvent(self, ev)
        if hasattr(self, "plots"):
            viewbox = self.plots[0].getViewBox()
            viewbox.autoRange()

    @QtCore.pyqtSlot(object)
    def mouseMoved(self, event):
        self.sandbox.cursor_tracker.sigMouseMoved.emit(event)


class ModelReferenceLayout(pg.GraphicsLayoutWidget):
    def __init__(self, sandbox, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)
        self.sandbox = sandbox

        self.plots = [
            DisplacementPlot(
                sandbox,
                title="Scene Displacement",
                component=lambda m: m.reference.scene.displacement,
            ),
            DisplacementPlot(
                sandbox,
                title="Model Residual",
                component=lambda m: m.reference.difference,
            ),
        ]
        self.plots[-1].addHintText()

        self._mov_sig = pg.SignalProxy(
            self.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved
        )

        for ip, plt in enumerate(self.plots):
            row = ip / 2
            col = ip % 2 + 1

            self.addItem(plt, row=row, col=col)
            plt.showGrid(x=True, y=True)
            plt.hideAxis("bottom")
            plt.hideAxis("left")
            plt.vb.border = pg.mkPen(50, 50, 50)
            if ip != 0:
                plt.setXLink(self.plots[0])
                plt.setYLink(self.plots[0])

        def getAxis(plt, orientation, label):
            axis = pg.AxisItem(orientation=orientation, linkView=plt.vb)
            axis.setLabel(label, units="m")
            return axis

        plts = self.plots
        self.addItem(getAxis(plts[0], "left", "Northing"), row=0, col=0)
        self.addItem(getAxis(plts[1], "left", "Northing"), row=1, col=0)
        self.addItem(getAxis(plts[0], "bottom", "Easting"), row=2, col=1)
        self.addItem(getAxis(plts[1], "bottom", "Easting"), row=2, col=2)

    @QtCore.pyqtSlot(object)
    def mouseMoved(self, event):
        self.sandbox.cursor_tracker.sigMouseMoved.emit(event)


class CursorRect(QtWidgets.QGraphicsRectItem):
    pen = pg.mkPen((0, 0, 0, 120), width=1.0)
    cursor = QtCore.QRectF((QtCore.QPointF(-1.5, -1.5)), (QtCore.QPointF(1.5, 1.5)))

    def __init__(self):
        QtWidgets.QGraphicsRectItem.__init__(self, self.cursor)
        self.setPen(self.pen)
        self.setZValue(1e9)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)


class DisplacementPlot(pg.PlotItem):
    def __init__(self, sandbox, component, title="Untitled"):
        pg.PlotItem.__init__(self)
        self.title = title
        self.sandbox = sandbox
        self.component = component

        self.cursor = CursorRect()
        self.addCursor()

        self.setAspectLocked(True)
        self.setLabels(bottom=("Easting", "m"), left=("Northing", "m"))

        border_pen = pg.mkPen(255, 255, 255, 50)

        self.image = pg.ImageItem(
            None, autoDownsample=False, border_pen=border_pen, useOpenGL=True
        )
        self.addItem(self.image)

        self.title_label = pg.LabelItem(
            text='<span style="color: #9E9E9E;">' "%s</span>" % self.title,
            justify="right",
            size="10pt",
            parent=self,
        )
        self.title_label.anchor(itemPos=(0.0, 0.0), parentPos=(0.01, 0.01))
        self.title_label.setOpacity(0.6)

        self.hint_text = None

        self.sandbox.sigModelUpdated.connect(self.update)
        self.sandbox.sources.modelAboutToBeReset.connect(self.removeSourceROIS)
        self.sandbox.sources.modelReset.connect(self.addSourceROIS)

        self.update()
        self.rois = []
        self.addSourceROIS()

    @property
    def data(self):
        return self.component(self.sandbox.model)

    @QtCore.pyqtSlot()
    def update(self):
        self.image.updateImage(self.data.T)
        self.transFromFrame()

    def transFromFrame(self):
        self.image.resetTransform()
        self.image.scale(self.sandbox.frame.dEmeter, self.sandbox.frame.dNmeter)

    @QtCore.pyqtSlot()
    def addSourceROIS(self):
        self.rois = []
        index = QtCore.QModelIndex()
        for isrc in range(self.sandbox.sources.rowCount(index)):
            index = self.sandbox.sources.index(isrc, 0, index)
            roi = index.data(SourceROI)
            roi.setParent(self)
            self.rois.append(roi)
            self.addItem(roi)

    @QtCore.pyqtSlot()
    def removeSourceROIS(self):
        if self.rois:
            for roi in self.rois:
                self.removeItem(roi)
            self.update()

    def addCursor(self):
        self.sandbox.cursor_tracker.sigCursorMoved.connect(self.drawCursor)
        self.sandbox.cursor_tracker.sigMouseMoved.connect(self.mouseMoved)
        self.addItem(self.cursor)

    @QtCore.pyqtSlot(object)
    def mouseMoved(self, event):

        if self.vb.sceneBoundingRect().contains(event[0]):
            map_pos = self.vb.mapSceneToView(event[0])

            img_pos = self.image.mapFromScene(event[0])
            pE, pN = img_pos.x(), img_pos.y()
            if (pE < 0 or pN < 0) or (
                pE > self.image.image.shape[0] or pN > self.image.image.shape[1]
            ):
                value = 0.0
            else:
                value = self.image.image[int(img_pos.x()), int(img_pos.y())]
            self.sandbox.cursor_tracker.sigCursorMoved.emit((map_pos, value))
            self.cursor.hide()
        else:
            if not getConfig().show_cursor:
                self.cursor.hide()
            else:
                self.cursor.show()

    @QtCore.pyqtSlot(object)
    def drawCursor(self, pos):
        pos, _ = pos
        self.cursor.setPos(pos)

    def addHintText(self):
        self.hint_text = pg.LabelItem(text="", justify="right", size="8pt", parent=self)
        self.hint_text.anchor(itemPos=(1.0, 1.0), parentPos=(1.0, 1.0))
        self.hint_text.text_template = (
            '<span style="font-family: monospace; color: #fff;'
            'background-color: #000;">'
            "East {0:08.2f} m | North {1:08.2f} m | "
            "Displacement {2:2.4f} m</span>"
        )
        self.hint_text.setOpacity(0.6)
        self.sandbox.cursor_tracker.sigCursorMoved.connect(self.updateHintText)

    def removeHintText(self):
        if self.hint_text is not None:
            self.removeItem(self.hint_text)
            self.hint_text = None

    @QtCore.pyqtSlot(object)
    def updateHintText(self, pos):
        pos, value = pos
        if self.hint_text is not None:
            self.hint_text.setText(
                self.hint_text.text_template.format(pos.x(), pos.y(), value)
            )


class DisplacementVectorPlot(DisplacementPlot):
    def __init__(self, *args, **kwargs):
        DisplacementPlot.__init__(self, *args, **kwargs)

        vectors = DisplacementVectors(self)
        self.addItem(vectors)


class DisplacementVectors(QtWidgets.QGraphicsItemGroup):
    def __init__(self, plot, *args, **kwargs):
        QtWidgets.QGraphicsItemGroup.__init__(self, parent=plot, *args, **kwargs)
        self.vb = plot.vb
        self.plot = plot
        self.image = plot.image
        self.sandbox = plot.sandbox

        self.scale_view = None
        self.scale_length = None
        self.vectors = []

        self.vb.geometryChanged.connect(self.prepareGeometryChange)
        getConfig().qconfig.updated.connect(self.createVectors)
        getConfig().qconfig.updated.connect(self.updateVectorAppearance)

        self.createVectors()

    def createVectors(self):
        for vec in self.vectors:
            vec.hide()

        if len(self.vectors) < getConfig().nvectors:
            while len(self.vectors) < getConfig().nvectors:
                vec = Vector(self)
                self.vectors.append(vec)
                self.addToGroup(vec)

    def updateVectorAppearance(self):
        Vector.arrow_color.setRgb(*getConfig().vector_color)
        Vector.arrow_brush.setColor(Vector.arrow_color)
        Vector.arrow_pen.setBrush(Vector.arrow_brush)
        Vector.arrow_pen.setWidth(getConfig().vector_pen_thickness)
        Vector.relative_length = getConfig().vector_relative_length

    def boundingRect(self):
        return self.vb.viewRect()

    def paint(self, painter, option, parent):
        r = self.vb.viewRect()
        h = r.height()
        w = r.width()

        nvectors = getConfig().nvectors

        nx = int(num.sqrt(nvectors) * float(w) / h)
        ny = int(num.sqrt(nvectors) * float(h) / w)
        dx = float(w) / nx
        dy = float(h) / ny
        d = dx if dx < dy else dy

        mat_N = self.sandbox.model.north.T
        mat_E = self.sandbox.model.east.T
        img_shape = self.image.image.shape
        ivec = 0

        length_scale = self.sandbox.model.max_horizontal_displacement
        self.length_scale = length_scale if length_scale > 0.0 else 1.0
        self.scale_view = (w + h) / 2 / painter.window().height() * 2.5

        for ix in range(nx):
            for iy in range(ny):
                if ivec > nvectors:
                    break
                vec = self.vectors[ivec]
                pos = QtCore.QPointF(r.x() + ix * dx + dx / 2, r.y() + iy * dy + dy / 2)

                # Slowest operation
                img_pos = self.plot.image.mapFromScene(self.vb.mapViewToScene(pos))

                pE = int(img_pos.x())
                pN = int(img_pos.y())

                if (
                    (pE >= img_shape[0] or pN >= img_shape[1])
                    or (pE < 0 or pN < 0)
                    or num.isnan(mat_E[pE, pN])
                    or num.isnan(mat_N[pE, pN])
                ):
                    dE = 0.0
                    dN = 0.0
                else:
                    dE = mat_E[pE, pN]
                    dN = mat_N[pE, pN]
                    dE = dE / self.length_scale * (d / self.scale_view)
                    dN = dN / self.length_scale * (d / self.scale_view)
                vec.setPos(pos)
                vec.setOrientation(dE, dN)

                if vec.scale() != self.scale_view:
                    vec.setScale(self.scale_view)
                vec.setVisible(True)

                ivec += 1

        while ivec < nvectors:
            self.vectors[ivec].hide()
            ivec += 1

        QtWidgets.QGraphicsItemGroup.paint(self, painter, option, parent)


class Vector(QtWidgets.QGraphicsItem):

    arrow_color = QtGui.QColor(*getConfig().vector_color)
    arrow_brush = QtGui.QBrush(arrow_color, QtCore.Qt.SolidPattern)
    arrow_pen = QtGui.QPen(
        arrow_brush, 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin
    )

    relative_length = getConfig().vector_relative_length

    def __init__(self, parent):
        QtWidgets.QGraphicsItem.__init__(self, parent=parent)

        self.p1 = QtCore.QPointF()
        self.p2 = QtCore.QPointF()
        self.line = QtCore.QLineF(self.p1, self.p2)

        self.setOrientation(0.0, 0.0)
        self.setZValue(10000)

    def boundingRect(self):
        return QtCore.QRectF(self.p1, self.p2)

    def setOrientation(self, dEast, dNorth):
        dEast *= self.relative_length / 100
        dNorth *= self.relative_length / 100
        self.p2.setX(dEast)
        self.p2.setY(dNorth)
        self.line.setP2(self.p2)

    def paint(self, painter, option, parent):
        if self.line.length() == 0.0:
            return
        painter.setPen(self.arrow_pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        arrow_length = self.line.length() * 0.3 * (self.relative_length / 100.0)
        d = self.line.angle()
        head_p1 = self.p2 - QtCore.QPointF(
            num.sin(d * d2r + num.pi / 3) * arrow_length,
            num.cos(d * d2r + num.pi / 3) * arrow_length,
        )

        head_p2 = self.p2 - QtCore.QPointF(
            num.sin(d * d2r + num.pi - num.pi / 3) * arrow_length,
            num.cos(d * d2r + num.pi - num.pi / 3) * arrow_length,
        )

        painter.drawLine(self.line)
        painter.drawPolyline(*[head_p1, self.p2, head_p2])


class ColormapPlots(pg.HistogramLUTWidget):
    def __init__(self):
        pg.HistogramLUTWidget.__init__(self, image=None)
        self.plots = []

        self.axis.setLabel("Displacement / m")

        zero_marker = pg.InfiniteLine(pos=0, angle=0, pen="w", movable=False)
        zero_marker.setValue(0.0)
        zero_marker.setZValue(1000)
        self.vb.addItem(zero_marker)

        self.axis.setLabel("Displacement / m")
        self.setSymColormap()

    @QtCore.pyqtSlot()
    def setSymColormap(self):
        cmap = {
            "ticks": [
                [0, (0, 0, 0, 255)],
                [1e-3, (106, 0, 31, 255)],
                [0.5, (255, 255, 255, 255)],
                [1.0, (8, 54, 104, 255)],
            ],
            "mode": "rgb",
        }
        cmap = {
            "ticks": [
                [0, (0, 0, 0)],
                [1e-3, (172, 56, 56)],
                [0.5, (255, 255, 255)],
                [1.0, (51, 53, 120)],
            ],
            "mode": "rgb",
        }

        lvl_min = lvl_max = 0.0
        for plot in self.plots:
            plt_min = num.nanmin(plot.data)
            plt_max = num.nanmax(plot.data)
            lvl_max = lvl_max if plt_max < lvl_max else plt_max
            lvl_min = lvl_min if plt_min > lvl_min else plt_min

        abs_range = max(abs(lvl_min), abs(lvl_max)) * 1.01

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


class SandboxSceneDockarea(dockarea.DockArea):
    def __init__(self, sandbox, *args, **kwargs):
        dockarea.DockArea.__init__(self)
        layout = SandboxSceneLayout(sandbox)

        cmap = ColormapPlots()
        for plt in layout.plots:
            cmap.addPlot(plt)
        sandbox.sigModelUpdated.connect(cmap.setSymColormap)

        cmap_dock = dockarea.Dock("Colormap", widget=cmap)
        cmap_dock.setStretch(1, None)

        layout_dock = dockarea.Dock("Model Sandbox", widget=layout)
        self.addDock(layout_dock, position="right")
        self.addDock(cmap_dock, position="right")


class ModelReferenceDockarea(dockarea.DockArea):
    def __init__(self, sandbox, *args, **kwargs):
        dockarea.DockArea.__init__(self)
        layout = ModelReferenceLayout(sandbox)

        cmap = ColormapPlots()
        for plt in layout.plots:
            cmap.addPlot(plt)

        cmap_dock = dockarea.Dock("Colormap", widget=cmap)
        cmap_dock.setStretch(1, None)

        layout_dock = dockarea.Dock("Model Sandbox", widget=layout)
        self.addDock(layout_dock, position="right")
        self.addDock(cmap_dock, position="right")
