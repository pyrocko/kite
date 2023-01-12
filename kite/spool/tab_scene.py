#!/usr/bin/python2
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from PyQt5 import QtCore, QtWidgets

from kite.qt_utils import loadUi

from .base import KiteParameterGroup, KitePlot, KiteView, get_resource

__all__ = ["KiteScene"]

km = 1e3
pen_roi = pg.mkPen((255, 23, 68), width=2)
pen_roi_highlight = pg.mkPen((115, 210, 22), width=2, style=QtCore.Qt.DashLine)


class PolygonMaskROI(pg.PolyLineROI):
    def _makePen(self):
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return pen_roi_highlight
        else:
            return self.pen


class KiteScene(KiteView):
    title = "Scene"

    def __init__(self, spool):
        model = spool.model
        self.model = model
        self.spool = spool

        scene_plot = KiteScenePlot(model)
        self.scene_plot = scene_plot
        self.main_widget = scene_plot
        self.tools = {
            # 'Components': KiteToolComponents(self.main_widget),
            # 'Displacement Transect': KiteToolTransect(self.main_widget),
        }

        self.param_scene = ParamScene(model, scene_plot)
        self.param_frame = ParamSceneFrame(model, expanded=False)
        self.param_meta = ParamSceneMeta(model, expanded=False)

        self.param_scene.addChild(self.param_frame)
        self.param_scene.addChild(self.param_meta)

        self.deramp_ctrl = DerampParams(model)

        self.parameters = [self.param_scene, self.deramp_ctrl]

        self.dialogTransect = KiteToolTransect(scene_plot, spool)

        spool.actionTransect.triggered.connect(self.dialogTransect.show)

        spool.actionAddPolygonMask.triggered.connect(scene_plot.newMaskPolygon)
        spool.actionTogglePolygonMask.setChecked(
            model.getScene().polygon_mask.is_enabled()
        )
        spool.actionTogglePolygonMask.toggled.connect(self.togglePolygonMask)

        KiteView.__init__(self)
        model.sigSceneModelChanged.connect(self.modelChanged)

    @QtCore.pyqtSlot()
    def modelChanged(self):
        self.main_widget.update()
        self.main_widget.transFromFrame()

        self.param_scene.updateValues()
        self.param_frame.updateValues()
        self.param_meta.updateValues()

        self.dialogTransect.close()

    def togglePolygonMask(self, checked):
        polygon_mask = self.model.scene.polygon_mask
        polygon_mask.set_enabled(checked)


class KiteScenePlot(KitePlot):
    def __init__(self, model):
        self.components_available = {
            "displacement": ["Scene.displacement", lambda sp: sp.scene.displacement],
            "theta": ["Scene.theta", lambda sp: sp.scene.theta],
            "phi": ["Scene.phi", lambda sp: sp.scene.phi],
            "thetaDeg": ["Scene.thetaDeg", lambda sp: sp.scene.thetaDeg],
            "phiDeg": ["Scene.phiDeg", lambda sp: sp.scene.phiDeg],
            "unitE": ["Scene.los.unitE", lambda sp: sp.scene.los.unitE],
            "unitN": ["Scene.los.unitN", lambda sp: sp.scene.los.unitN],
            "unitU": ["Scene.los.unitU", lambda sp: sp.scene.los.unitU],
        }

        if model.scene.displacement_px_var is not None:
            self.components_available["displacement_px_var"] = [
                "Scene.displacement_px_var",
                lambda sp: sp.scene.displacement_px_var,
            ]

        self._component = "displacement"

        KitePlot.__init__(self, model=model, los_arrow=True)

        model.sigFrameChanged.connect(self.onFrameChange)
        model.sigSceneModelChanged.connect(self.update)
        self.loadMaskPolygons()

        # Todo: use activateView
        model.sigSceneChanged.connect(self.update)
        self.model = model

    def roiToVertices(self, roi):
        frame = self.model.scene.frame
        return [
            (h.pos().x() / frame.dE, h.pos().y() / frame.dN) for h in roi.getHandles()
        ]

    def verticesToRoi(self, vertices):
        frame = self.model.scene.frame
        return [
            ((v[0] / frame.cols) * frame.lengthE, (v[1] / frame.rows) * frame.lengthN)
            for v in vertices
        ]

    def loadMaskPolygons(self):
        scene = self.model.scene
        for pid, vertices in scene.polygon_mask.polygons.items():
            self.createMaskPolygon(pid, vertices)

    def newMaskPolygon(self):
        scene = self.model.scene
        frame = self.model.scene.frame

        vertices = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        vertices[:, 0] *= frame.cols / 6
        vertices[:, 1] *= frame.rows / 6
        vertices[:, 0] += frame.cols / 2
        vertices[:, 1] += frame.rows / 2

        pid = scene.polygon_mask.add_polygon(vertices.tolist())
        self.createMaskPolygon(pid, vertices)

    def createMaskPolygon(self, pid, vertices):
        roi = PolygonMaskROI(
            self.verticesToRoi(vertices),
            pen=pen_roi,
            movable=False,
            removable=True,
            closed=True,
        )
        roi.pid = pid

        roi.sigRegionChangeFinished.connect(self.updatePolygonMask)
        roi.sigRemoveRequested.connect(self.removePolygonMask)
        self.addItem(roi)

    def updatePolygonMask(self, roi):
        vertices = self.roiToVertices(roi)
        self.model.scene.polygon_mask.update_polygon(roi.pid, vertices)

    def removePolygonMask(self, roi):
        self.model.scene.polygon_mask.remove_polygon(roi.pid)
        self.removeItem(roi)

    def onFrameChange(self):
        self.update()
        self.transFromFrame()


class KiteToolTransect(QtWidgets.QDialog):
    def __init__(self, plot, parent=None):
        QtWidgets.QDialog.__init__(self, parent)

        loadUi(get_resource("transect.ui"), baseinstance=self)

        icon = self.style().standardIcon

        self.closeButton.setIcon(icon(QtWidgets.QStyle.SP_DialogCloseButton))
        self.createButton.setIcon(icon(QtWidgets.QStyle.SP_ArrowUp))
        self.removeButton.setIcon(icon(QtWidgets.QStyle.SP_DialogDiscardButton))

        self.plot = plot
        self.poly_line = None

        self.trans_plot = pg.PlotDataItem(
            antialias=True, fillLevel=0.0, fillBrush=pg.mkBrush(0, 127, 0, 150)
        )

        self.plt_wdgt = pg.PlotWidget()
        self.plt_wdgt.setLabels(bottom={"Distance", " m"}, left="Displacement [m]")

        self.plt_wdgt.showGrid(True, True, alpha=0.5)
        self.plt_wdgt.enableAutoRange()
        self.plt_wdgt.addItem(self.trans_plot)

        self.layoutPlot.addWidget(self.plt_wdgt)
        self.plot.image.sigImageChanged.connect(self.updateTransPlot)
        self.createButton.released.connect(self.addPolyLine)
        self.removeButton.released.connect(self.removePolyLine)

        parent.model.sigConfigChanged.connect(self.close)

    def addPolyLine(self):
        [[xmin, xmax], [ymin, ymax]] = self.plot.viewRange()
        self.poly_line = pg.PolyLineROI(
            positions=[
                (xmin + (xmax - xmin) * 0.4, ymin + (ymax - ymin) * 0.4),
                (xmin + (xmax - xmin) * 0.6, ymin + (ymax - ymin) * 0.6),
            ],
            pen=pg.mkPen("g", width=2),
        )
        self.plot.addItem(self.poly_line)
        self.updateTransPlot()

        self.poly_line.sigRegionChangeFinished.connect(self.updateTransPlot)

    def removePolyLine(self):
        if self.poly_line is None:
            return

        self.plot.removeItem(self.poly_line)
        self.poly_line = None
        self.updateTransPlot()

    def closeEvent(self, event):
        self.removePolyLine()

    def updateTransPlot(self):
        if self.poly_line is None:
            return

        transect = np.ndarray((0))
        length = 0
        for line in self.poly_line.segments:
            transect = np.append(
                transect, line.getArrayRegion(self.plot.image.image, self.plot.image)
            )
            p1, p2 = line.listPoints()
            length += (p2 - p1).length()
        # interpolate over NaNs
        nans, x = np.isnan(transect), lambda z: z.nonzero()[0]
        transect[nans] = np.interp(x(nans), x(~nans), transect[~nans])
        length = np.linspace(0, length, transect.size)

        self.trans_plot.setData(length, transect)
        self.plt_wdgt.setLimits(
            xMin=length.min(),
            xMax=length.max(),
            yMin=transect.min(),
            yMax=transect.max() * 1.1,
        )
        return


class ParamScene(KiteParameterGroup):
    def __init__(self, model, plot, **kwargs):
        kwargs["type"] = "group"
        kwargs["name"] = "Scene"
        self.plot = plot

        self.parameters = {
            "min value": lambda plot: np.nanmin(plot.data),
            "max value": lambda plot: np.nanmax(plot.data),
            "mean value": lambda plot: np.nanmean(plot.data),
        }

        self.plot.image.sigImageChanged.connect(self.updateValues)

        KiteParameterGroup.__init__(self, model=self.plot, model_attr=None, **kwargs)

        def changeComponent(parameter):
            self.plot.component = parameter.value()

        p = {
            "name": "display",
            "values": {
                "displacement": "displacement",
                "theta": "theta",
                "phi": "phi",
                "thetaDeg": "thetaDeg",
                "phiDeg": "phiDeg",
                "los.unitE": "unitE",
                "los.unitN": "unitN",
                "los.unitU": "unitU",
            },
            "value": "displacement",
            "tip": "Change the displayed component of the displacement field.",
        }

        if model.scene.displacement_px_var is not None:
            p["values"]["displacement_px_var"] = "displacement_px_var"

        component = pTypes.ListParameter(**p)
        component.sigValueChanged.connect(changeComponent)
        self.pushChild(component)


class ParamSceneFrame(KiteParameterGroup):
    def __init__(self, model, **kwargs):
        kwargs["type"] = "group"
        kwargs["name"] = ".frame"

        self.parameters = OrderedDict(
            [
                ("cols", None),
                ("rows", None),
                ("dN", None),
                ("dE", None),
                ("spacing", None),
                ("llLat", None),
                ("llLon", None),
                ("llNutm", None),
                ("llEutm", None),
                ("utm_zone", None),
                ("utm_zone_letter", None),
            ]
        )

        model.sigFrameChanged.connect(self.updateValues)

        KiteParameterGroup.__init__(self, model=model, model_attr="frame", **kwargs)


class ParamSceneMeta(KiteParameterGroup):
    def __init__(self, model, **kwargs):
        from datetime import datetime as dt

        kwargs["type"] = "group"
        kwargs["name"] = ".meta"

        def str_to_time(d, fmt="%Y-%m-%d %H:%M:%S"):
            return dt.strftime(dt.fromtimestamp(d), fmt)

        self.parameters = OrderedDict(
            [
                ("time_master", lambda sc: str_to_time(sc.meta.time_master)),
                ("time_slave", lambda sc: str_to_time(sc.meta.time_slave)),
                ("time_separation", lambda sc: "%s" % sc.meta.time_separation),
            ]
        )

        model.sigConfigChanged.connect(self.updateValues)

        KiteParameterGroup.__init__(self, model=model, model_attr="scene", **kwargs)

        def update_meta_info(key, value):
            self.model.scene.meta.__setattr__(key, value)

        p = {
            "name": "scene_title",
            "value": self.model.scene.meta.scene_title,
            "type": "str",
            "tip": "Title of the displacement scene",
        }

        self.scene_title = pTypes.SimpleParameter(**p)
        self.scene_title.sigValueChanged.connect(
            lambda v: update_meta_info("scene_title", v.value())
        )

        p = {"name": "scene_id", "value": self.model.scene.meta.scene_id, "type": "str"}

        self.scene_id = pTypes.SimpleParameter(**p)
        self.scene_id.sigValueChanged.connect(
            lambda v: update_meta_info("scene_id", v.value())
        )

        p = {
            "name": "satellite_name",
            "value": self.model.scene.meta.satellite_name,
            "type": "str",
            "tip": "Name of the satellite",
        }

        self.satellite_name = pTypes.SimpleParameter(**p)
        self.satellite_name.sigValueChanged.connect(
            lambda v: update_meta_info("satellite_name", v.value())
        )

        p = {
            "name": "orbital_node",
            "values": {
                "Ascending": "Ascending",
                "Descending": "Descending",
                "Undefined": "Undefined",
            },
            "value": self.model.scene.meta.orbital_node,
            "tip": "Satellite orbit direction",
        }
        self.orbital_node = pTypes.ListParameter(**p)
        self.orbital_node.sigValueChanged.connect(
            lambda v: update_meta_info("orbital_node", v.value())
        )

        self.pushChild(self.orbital_node)
        self.pushChild(self.satellite_name)
        self.pushChild(self.scene_id)
        self.pushChild(self.scene_title)


class DerampParams(KiteParameterGroup):
    def __init__(self, model, **kwargs):
        scene = model.getScene()
        kwargs["type"] = "group"
        kwargs["name"] = "Scene.detrend"

        KiteParameterGroup.__init__(self, model=model, model_attr="scene", **kwargs)

        p = {
            "name": "demean",
            "type": "bool",
            "value": scene.deramp.config.demean,
            "tip": "subtract mean of displacement",
        }
        self.demean = pTypes.SimpleParameter(**p)

        def toggle_demean(param, checked):
            self.model.getScene().deramp.set_demean(checked)

        self.demean.sigValueChanged.connect(toggle_demean)

        p = {
            "name": "applied",
            "type": "bool",
            "value": scene.deramp.config.applied,
            "tip": "detrend the scene",
        }
        self.applied = pTypes.SimpleParameter(**p)

        def toggle_applied(param, checked):
            self.model.getScene().deramp.set_enabled(checked)

        self.applied.sigValueChanged.connect(toggle_applied)

        self.pushChild(self.applied)
        self.pushChild(self.demean)
