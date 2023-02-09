from os import path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from PyQt5 import QtCore
from pyqtgraph import dockarea

from kite.qt_utils import _viridis_data
from kite.util import calcPrecision, formatScalar

__all__ = ["KiteView", "KitePlot", "KiteToolColormap", "KiteParameterGroup"]


d2r = np.pi / 180.0


def get_resource(filename):
    return path.join(path.dirname(path.realpath(__file__)), "res", filename)


class KiteView(dockarea.DockArea):
    title = "View Prototype"

    def __init__(self):
        dockarea.DockArea.__init__(self)
        self.tool_docks = []
        self.tools = getattr(self, "tools", {})

        self.colormap = KiteToolColormap(self.main_widget)

        dock_main = dockarea.Dock(
            self.title, autoOrientation=False, widget=self.main_widget
        )
        dock_colormap = dockarea.Dock(
            "Colormap", autoOrientation=False, widget=self.colormap
        )
        dock_colormap.setStretch(1, None)

        for i, (name, tool) in enumerate(self.tools.items()):
            self.tool_docks.append(
                dockarea.Dock(name, widget=tool, size=(2, 2), autoOrientation=False)
            )
            self.addDock(self.tool_docks[-1], position="bottom")

        self.addDock(dock_main, position="left")
        self.addDock(dock_colormap, position="right")

    @QtCore.pyqtSlot()
    def activateView(self):
        pass

    @QtCore.pyqtSlot()
    def deactivateView(self):
        pass


class KitePlotThread(QtCore.QThread):
    completeSignal = QtCore.pyqtSignal()

    def __init__(self, plot):
        super().__init__()
        self.plot = plot

    def run(self):
        plot = self.plot
        plot._data = plot.components_available[plot.component][1](plot.model)
        self.completeSignal.emit()


class LOSArrow(pg.GraphicsWidget, pg.GraphicsWidgetAnchor):
    def __init__(self, model):
        pg.GraphicsWidget.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)

        self.model = model

        self.arrow = pg.ArrowItem(
            parent=self,
            angle=0.0,
            brush=(0, 0, 0, 180),
            pen=(255, 255, 255),
            pxMode=True,
        )

        self.label = pg.LabelItem(
            "Towards Sat.", justify="right", size="8pt", parent=self
        )
        self.label.anchor(itemPos=(1.0, -1.0), parentPos=(1.0, 0.0))
        # self.label.setBrush(pg.mkBrush(255, 255, 255, 180))
        # self.label.setFont(QtGui.QFont(
        #     "Helvetica", weight=QtGui.QFont.DemiBold))

        self.orientArrow()
        self.model.sigSceneChanged.connect(self.orientArrow)
        self.setFlag(self.ItemIgnoresTransformations)

    @QtCore.pyqtSlot()
    def orientArrow(self):
        phi = np.nanmedian(self.model.scene.phi)
        theta = np.nanmedian(self.model.scene.theta)

        angle = 180.0 - np.rad2deg(phi)

        theta_f = theta / (np.pi / 2)

        tipAngle = 30.0 + theta_f * 20.0
        tailLen = 15 + theta_f * 15.0

        self.arrow.setStyle(
            angle=0.0, tipAngle=tipAngle, tailLen=tailLen, tailWidth=6, headLen=25
        )
        self.arrow.setRotation(angle)

        rect_label = self.label.boundingRect()
        self.arrow.boundingRect()

        self.label.setPos(-rect_label.width() / 2.0, rect_label.height() * 1.33)

    def setParentItem(self, parent):
        pg.GraphicsWidget.setParentItem(self, parent)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())


class KitePlot(pg.PlotWidget):
    def __init__(self, model, los_arrow=False, auto_downsample=False):
        pg.PlotWidget.__init__(self)
        self.model = model
        self.draw_time = 0.0
        self._data = None
        self.updateThread = KitePlotThread(self)
        self.updateThread.completeSignal.connect(self._updateImageFromData)

        border_pen = pg.mkPen(255, 255, 255, 50)
        self.image = pg.ImageItem(
            None, autoDownsample=auto_downsample, border=border_pen, useOpenGL=True
        )

        self.hillshade = None
        self.elevation = None

        self.setAspectLocked(True)
        self.plotItem.getAxis("left").setZValue(100)
        self.plotItem.getAxis("bottom").setZValue(100)

        self.hint = {
            "east": 0.0,
            "north": 0.0,
            "value": np.nan,
            "measure": self.component.title(),
            "vlength": "03",
            "precision": "3",
        }

        self.hint_text = pg.LabelItem(
            text="", justify="right", size="8pt", parent=self.plotItem
        )
        self.hint_text.anchor(itemPos=(1.0, 0.0), parentPos=(1.0, 0.0))
        self.hint_text.setOpacity(0.6)

        if self.model.frame.isMeter():
            self.hint_text.text_template = (
                '<span style="font-family: monospace; color: #fff;'
                'background-color: #000;">'
                "East {east:08.2f} m | North {north:08.2f} m |"
                " {measure} {value:{length}.{precision}f}</span>"
            )
            self.setLabels(bottom=("Easting", "m"), left=("Northing", "m"))
        elif self.model.frame.isDegree():
            self.hint_text.text_template = (
                '<span style="font-family: monospace; color: #fff;'
                'background-color: #000;">'
                "Lon {east:03.3f}&deg; | Lat {north:02.3f}&deg; |"
                " {measure} {value:{length}.{precision}f}</span>"
            )
            self.setLabels(bottom="Longitude", left="Latitude")

        self.addItem(self.image)
        self.update()

        self.transFromFrame()
        self._move_sig = pg.SignalProxy(
            self.image.scene().sigMouseMoved, rateLimit=25, slot=self.mouseMoved
        )

        if los_arrow:
            self.addLOSArrow()

        # self.addIsocurve()
        # self.scalebar()

    def addLOSArrow(self):
        self.los_arrow = LOSArrow(self.model)
        self.los_arrow.setParentItem(self.graphicsItem())
        self.los_arrow.anchor(
            itemPos=(1.0, 0.0), parentPos=(1, 0.0), offset=(-10.0, 40.0)
        )

    def enableHillshade(self):
        from scipy.signal import fftconvolve

        frame = self.model.frame
        scene = self.model.scene

        elevation = scene.get_elevation()

        contrast = 1.0

        size_ramp = 10
        ramp = np.linspace(-1, 1, size_ramp)[::-1]

        ramp_x = np.tile(ramp, size_ramp).reshape(size_ramp, size_ramp)
        ramp_y = ramp_x.T
        ramp = ramp_x + ramp_y

        ramp = ramp / ramp.max() * contrast
        # ramp = np.array([[-1, -.5, 0], [0, .5, 1.]]) * contrast
        # convolution of two 2-dimensional arrays

        shad = fftconvolve(elevation, ramp, mode="valid")
        shad = -1.0 * shad

        # if there are strong artificial edges in the data, shades get
        # dominated by them. Cutting off the largest and smallest 2% of
        # shades helps

        percentile2 = np.quantile(shad, 0.02)
        percentile98 = np.quantile(shad, 0.98)

        shad = np.where(shad > percentile98, percentile98, shad)
        shad = np.where(shad < percentile2, percentile2, shad)
        shad -= shad.min()
        shad /= shad.max()

        # normalize to range [0 1]
        hillshade = pg.ImageItem(
            None, autoDownsample=False, border=None, useOpenGL=False
        )
        hillshade.resetTransform()
        hillshade.scale(frame.dE, frame.dN)

        elev_img = pg.ImageItem(
            None, autoDownsample=False, border=None, useOpenGL=False
        )
        elev_img.resetTransform()
        elev_img.scale(frame.dE, frame.dN)

        elev_img.updateImage(elevation.T, autoLevels=True)
        hillshade.updateImage(shad.T, autoLevels=True)

        self.hillshade = hillshade
        self.elevation = elev_img
        self.removeItem(self.image)
        self.addItem(self.elevation)
        self.addItem(self.hillshade)
        self.addItem(self.image)

        self.update()

    def disableHillshade(self):
        if not self.hillshade:
            return
        self.removeItem(self.hillshade)
        self.hillshade = None

    def transFromFrame(self):
        frame = self.model.frame

        self.image.resetTransform()
        self.image.scale(frame.dE, frame.dN)

    def scalebar(self):
        """Not working"""
        self.scale_bar = pg.ScaleBar(10, width=5, suffix="m")
        self.scale_bar.setParentItem(self.plotItem)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-20, -20))

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        if component not in self.components_available.keys():
            raise AttributeError("Invalid component %s" % component)
        self._component = component
        self.update()

    @property
    def data(self):
        if self._data is None:
            self._data = self.components_available[self.component][1](self.model)
        return self._data
        # return self._data  # np.nan_to_num(_data)

    @QtCore.pyqtSlot()
    def update(self, obj=None):
        if not self.updateThread.isRunning():
            self.updateThread.start()

    @QtCore.pyqtSlot()
    def _updateImageFromData(self):
        self.image.updateImage(self.data.T, opacity=0.7 if self.hillshade else 1.0)

        self.hint["precision"], self.hint["vlength"] = calcPrecision(self.data)
        self.mouseMoved()

    @QtCore.pyqtSlot(object)
    def mouseMoved(self, event=None):
        frame = self.model.frame
        if event is None:
            return
        elif self.image.sceneBoundingRect().contains(event[0]):
            map_pos = self.plotItem.vb.mapSceneToView(event[0])
            if not map_pos.isNull():
                img_pos = self.image.mapFromScene(*event)
                value = self.image.image[int(img_pos.x()), int(img_pos.y())]

                self.hint["east"] = map_pos.x()
                self.hint["north"] = map_pos.y()
                self.hint["value"] = value

                if frame.isDegree():
                    self.hint["east"] += frame.llLon
                    self.hint["north"] += frame.llLat

        self.hint["length"] = (
            "03" if np.isnan(self.hint["value"]) else self.hint["vlength"]
        )
        self.hint_text.setText(self.hint_text.text_template.format(**self.hint))


class KiteToolColormap(pg.HistogramLUTWidget):
    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)
        self._plot = plot
        self.prev_levels = None
        self.symmetric_colormap = True

        zero_marker = pg.InfiniteLine(pos=0, angle=0, pen="w", movable=False)
        zero_marker.setValue(0.0)
        zero_marker.setZValue(1000)
        self.vb.addItem(zero_marker)

        self.axis.setLabel("Displacement / m")
        # self.plot.rotate(-90)
        # self.layout.rotate(90)
        # self.gradient.setOrientation('bottom')
        self.setSymColormap()
        self._plot.image.sigImageChanged.connect(self.imageChanged)

        self.sigLevelsChanged.connect(self.symmetricLevels)
        # self.isoCurveControl()

    def set_symmetric_colormap(self, symmetric):
        self.symmetric_colormap = symmetric

    @QtCore.pyqtSlot()
    def imageChanged(self):
        if self._plot.component == "weight":
            self.setQualitativeColormap()
        else:
            self.setSymColormap()

        max_range = np.nanmax(np.abs(self._plot.data))
        if max_range is not np.nan:
            self.vb.setYRange(-max_range, max_range)

    @QtCore.pyqtSlot(object)
    def symmetricLevels(self, *args):
        if not self.symmetric_colormap:
            return

        levels = self.getLevels()
        if self.prev_levels is None:
            self.prev_levels = levels

        min_changed = bool(self.prev_levels[0] - levels[0])
        func = min if min_changed else max

        max_lvl = abs(func(levels))
        self.prev_levels = levels
        self.setLevels(-max_lvl, max_lvl)

    def setSymColormap(self):
        cmap = {
            "ticks": [
                [0.0, (0, 0, 0, 255)],
                [1e-3, (106, 0, 31, 255)],
                [0.5, (255, 255, 255, 255)],
                [1.0, (8, 54, 104, 255)],
            ],
            "mode": "rgb",
        }
        cmap = {
            "ticks": [
                [0.0, (0, 0, 0)],
                [1e-3, (172, 56, 56)],
                [0.5, (255, 255, 255)],
                [1.0, (51, 53, 120)],
            ],
            "mode": "rgb",
        }

        relevant_data = np.abs(self._plot.data[np.isfinite(self._plot.data)])
        if np.any(relevant_data):
            lvl_max = np.quantile(relevant_data, 0.999)
        else:
            lvl_max = 1.0

        self.gradient.restoreState(cmap)
        self.setLevels(-lvl_max, lvl_max)

    def setQualitativeColormap(self):
        nc = len(_viridis_data) - 1
        cmap = {"mode": "rgb"}
        cmap["ticks"] = [[float(i) / nc, c] for i, c in enumerate(_viridis_data)]
        self.gradient.restoreState(cmap)
        self.setLevels(np.nanmin(self._plot.data), np.nanmax(self._plot.data))

    def isoCurveControl(self):
        iso_ctrl = pg.InfiniteLine(pos=0, angle=0, pen="g", movable=True)
        iso_ctrl.setValue(0.0)
        iso_ctrl.setZValue(1000)

        def isolineChange():
            self._plot.iso.setLevel(iso_ctrl.value())

        iso_ctrl.sigDragged.connect(isolineChange)
        self.vb.addItem(iso_ctrl)


class KiteParameterGroup(pTypes.GroupParameter):
    def __init__(self, model, model_attr=None, **kwargs):
        self.model = model
        self.model_attr = model_attr

        self.parameters = getattr(self, "parameters", [])

        if isinstance(self.parameters, list):
            self.parameters = dict.fromkeys(self.parameters)

        pTypes.GroupParameter.__init__(self, **kwargs)
        self.updateValues()

    def updateValues(self):
        if not hasattr(self, "parameters"):
            return

        if self.model_attr is not None and hasattr(self.model, self.model_attr):
            model = getattr(self.model, self.model_attr)
        else:
            model = self.model

        for param, f in self.parameters.items():
            QtCore.QCoreApplication.processEvents()
            try:
                if callable(f):
                    value = f(model)
                    # print('Updating %s: %s (func)' % (param, value))
                else:
                    value = getattr(model, param)
                    # print('Updating %s: %s (getattr)' % (param, value))
                try:
                    value = formatScalar(float(value))
                except ValueError:
                    pass
            except AttributeError:
                value = "n/a"
            try:
                self.child(param).setValue(value)
            except Exception:
                self.addChild(
                    {"name": param, "value": value, "type": "str", "readonly": True}
                )

    def pushChild(self, child, **kwargs):
        self.insertChild(0, child, **kwargs)
