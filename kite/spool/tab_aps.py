import numpy as np
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from PyQt5 import QtCore, QtWidgets
from pyqtgraph import dockarea

from kite.qt_utils import loadUi

from .base import KiteParameterGroup, KitePlot, KiteView, get_resource
from .tab_covariance import KiteSubplot

km = 1e3
pen_roi = pg.mkPen((78, 154, 6), width=2)
pen_roi_highlight = pg.mkPen((115, 210, 22), width=2, style=QtCore.Qt.DashLine)

pen_aps = pg.mkPen((255, 255, 255, 100), width=1.25)
brush_aps = pg.mkBrush((255, 255, 255, 100))
pen_aps_model = pg.mkPen((204, 0, 0), width=2, style=QtCore.Qt.DotLine)


class KiteAPS(KiteView):
    title = "Scene.APS"

    def __init__(self, spool):
        model = spool.model
        self.model = model

        self.main_widget = KiteAPSPlot(model)
        self.aps_correlation = KiteAPSCorrelation(self.main_widget)

        self.tools = {"APS Correlation": self.aps_correlation}

        self.aps_ctrl = EmpiricalAPSParams(model)
        self.parameters = [self.aps_ctrl]

        KiteView.__init__(self)

        for dock in self.tool_docks:
            dock.setStretch(10, 0.5)

        spool.actionApplyEmpiricalAPS.setChecked(model.getScene().aps.is_enabled())
        spool.actionApplyEmpiricalAPS.toggled.connect(self.toggleAPS)

        self.main_widget.region_changed.connect(self.aps_correlation.update)

        self.GACOSDialog = GACOSCorrectionDialog(model, spool)
        spool.actionGACOS.triggered.connect(self.GACOSDialog.show)

    @QtCore.pyqtSlot()
    def activateView(self):
        self.aps_correlation.activatePlot()
        self.main_widget.activatePlot()

    @QtCore.pyqtSlot()
    def deactivateView(self):
        self.aps_correlation.deactivatePlot()
        self.main_widget.activatePlot()

    def toggleAPS(self, checked):
        if checked:
            msg = QtWidgets.QMessageBox.question(
                self,
                "Apply APS to Scene",
                "Are you sure you want to apply APS to the scene?",
            )
            if msg == QtWidgets.QMessageBox.StandardButton.Yes:
                self.model.getScene().aps.set_enabled(True)
        else:
            self.model.getScene().aps.set_enabled(False)


class KiteAPSPlot(KitePlot):
    region_changed = QtCore.pyqtSignal()

    class TopoPatchROI(pg.RectROI):
        def _makePen(self):
            # Generate the pen color for this ROI based on its current state.
            if self.mouseHovering:
                return pen_roi_highlight
            else:
                return self.pen

    def __init__(self, model):
        self.components_available = {
            "displacement": ["Displacement", lambda sp: sp.scene.displacement]
        }
        self._component = "displacement"

        KitePlot.__init__(self, model=model, los_arrow=False, auto_downsample=True)

        llE, llN, sizeE, sizeN = self.model.aps.get_patch_coords()
        self.roi = self.TopoPatchROI(
            pos=(llE, llN), size=(sizeE, sizeN), sideScalers=True, pen=pen_roi
        )

        self.roi.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.roi.sigRegionChangeFinished.connect(self.updateTopoRegion)

        self.addItem(self.roi)

    @QtCore.pyqtSlot()
    def updateTopoRegion(self):
        # data = self.roi.getArrayRegion(self.image.image, self.image)
        # data[data == 0.] = np.nan
        # if np.all(np.isnan(data)):
        #     return

        llE, llN = self.roi.pos()
        sizeE, sizeN = self.roi.size()
        patch_coords = (llE, llN, sizeE, sizeN)
        self.model.aps.set_patch_coords(*patch_coords)
        self.region_changed.emit()

    def activatePlot(self):
        self.enableHillshade()
        self.model.sigSceneChanged.connect(self.update)

    def deactivatePlot(self):
        self.model.sigSceneChanged.disconnect(self.update)


class KiteAPSCorrelation(KiteSubplot):
    MAXPOINTS = 10000

    def __init__(self, parent_plot):
        KiteSubplot.__init__(self, parent_plot)

        self.aps_correlation = pg.ScatterPlotItem(
            antialias=True, brush=brush_aps, pen=pen_aps, size=4
        )

        self.aps_model = pg.PlotDataItem(antialias=True, pen=pen_aps_model)

        self.legend = pg.LegendItem(offset=(0.0, 0.5))

        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.aps_model, "")

        self.addItem(self.aps_correlation)
        self.addItem(self.aps_model)

        self.plot.setLabels(bottom="Elevation (m)", left="Displacement (m)")

    @QtCore.pyqtSlot()
    def update(self):
        aps = self.model.aps
        elevation, displacement = aps.get_data()

        step = max(1, displacement.size // self.MAXPOINTS)

        self.aps_correlation.setData(elevation[::step], displacement[::step])

        slope, intercept = aps.get_correlation()
        elev = np.array([elevation.min(), elevation.max()])
        model = elev * slope + intercept

        self.aps_model.setData(elev, model)

        self.legend.items[-1][1].setText("Slope %.4f m / km" % (slope * km))

    def activatePlot(self):
        self.model.sigSceneChanged.connect(self.update)
        self.update()

    def deactivatePlot(self):
        self.model.sigSceneChanged.disconnect(self.update)


class GACOSCorrectionDialog(QtWidgets.QDialog):
    class GACOSPlot(KitePlot):
        def __init__(self, model, parent):
            self._component = "gacos_correction"
            self.parent = parent

            KitePlot.__init__(self, model)
            self.model = model

            self.update()

        def update(self):
            gacos = self.model.scene.gacos
            if not gacos.has_data():
                return

            corr = gacos.get_correction()
            self.image.updateImage(corr.T, autoLevels=True)
            self.transFromFrame()

    def __init__(self, model, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.model = model

        loadUi(get_resource("gacos_correction.ui"), baseinstance=self)
        self.closeButton.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
        )

        self.gacos_plot = self.GACOSPlot(model, self)
        self.dockarea = dockarea.DockArea(self)

        self.dockarea.addDock(
            dockarea.Dock(
                "GACOS.get_correction()",
                widget=self.gacos_plot,
                size=(4, 4),
                autoOrientation=False,
            ),
            position="left",
        )

        self.horizontalLayoutPlot.addWidget(self.dockarea)
        self.loadGrids.released.connect(self.load_grids)
        self.toggleGACOS.released.connect(self.toggle_gacos)
        self.update_widgets()

    @QtCore.pyqtSlot()
    def load_grids(self):
        gacos = self.model.scene.gacos

        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            filter="GACOS file *.ztd (*.ztd)", caption="Load two GACOS APS Grids"
        )
        if not filenames:
            return
        if len(filenames) != 2:
            QtWidgets.QMessageBox.warning(
                self,
                "GACOS APS Correction Error",
                "We need two GACOS Grids to perform APS correction!",
            )
            return

        try:
            for filename in filenames:
                gacos.load(filename)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "GACOS APS Correction Error", str(e))
            return
        self.gacos_plot.update()
        self.update_widgets()

    def update_widgets(self):
        gacos = self.model.scene.gacos

        tmpl_str = """
<b>Loaded GACOS Grids</b><br />
Grid 1 @{grd0.time}: {grd0.filename}<br />
Grid 2 @{grd1.time}: {grd1.filename}
"""

        if gacos.has_data():
            self.toggleGACOS.setEnabled(True)

            self.gridInformation.setText(
                tmpl_str.format(grd0=gacos.grids[0], grd1=gacos.grids[1])
            )

            if gacos.is_enabled():
                self.toggleGACOS.setText("Remove GACOS APS")
            else:
                self.toggleGACOS.setText("Apply GACOS APS")
        else:
            self.gridInformation.setText("No grids loaded.")
            self.toggleGACOS.setEnabled(False)

    def toggle_gacos(self):
        gacos = self.model.scene.gacos
        if gacos.is_enabled():
            gacos.set_enabled(False)
        else:
            gacos.set_enabled(True)
        self.update_widgets()


class EmpiricalAPSParams(KiteParameterGroup):
    def __init__(self, model, **kwargs):
        scene = model.getScene()
        kwargs["type"] = "group"
        kwargs["name"] = "Scene.APS (empirical)"

        KiteParameterGroup.__init__(self, model=model, model_attr="scene", **kwargs)

        p = {
            "name": "applied",
            "type": "bool",
            "value": scene.aps.config.applied,
            "tip": "detrend the scene",
        }
        self.applied = pTypes.SimpleParameter(**p)

        def toggle_applied(param, checked):
            self.model.getScene().aps.set_enabled(checked)

        self.applied.sigValueChanged.connect(toggle_applied)

        self.pushChild(self.applied)
