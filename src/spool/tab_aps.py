import numpy as num
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes

from PyQt5 import QtGui, QtCore

from .base import KiteView, KitePlot
from .tab_covariance import KiteSubplot


km = 1e3
pen_roi = pg.mkPen(
    (78, 154, 6), width=2)
pen_roi_highlight = pg.mkPen(
    (115, 210, 22), width=2, style=QtCore.Qt.DashLine)

pen_aps = pg.mkPen(
    (255, 255, 255, 100), width=1.25)
brush_aps = pg.mkBrush(
    (255, 255, 255, 100))
pen_aps_model = pg.mkPen(
    (204, 0, 0), width=2, style=QtCore.Qt.DotLine)


class KiteAPS(KiteView):
    title = 'Scene.APS'

    def __init__(self, spool):
        model = spool.model
        self.model = model

        self.main_widget = KiteAPSPlot(model)

        self.aps_correlation = KiteAPSCorrelation(self.main_widget)

        self.tools = {
            'APS Correlation': self.aps_correlation
        }

        KiteView.__init__(self)

        for dock in self.tool_docks:
            dock.setStretch(10, .5)

        spool.actionApply_APS.triggered.connect(self.applyAPS)
        spool.actionRemove_APS.triggered.connect(self.removeAPS)
        spool.menu_APS.setEnabled(True)

    @QtCore.pyqtSlot()
    def activateView(self):
        self.aps_correlation.activatePlot()
        self.main_widget.activatePlot()

    @QtCore.pyqtSlot()
    def deactivateView(self):
        self.aps_correlation.deactivatePlot()
        self.main_widget.activatePlot()

    def applyAPS(self):
        msg = QtGui.QMessageBox.question(
            self,
            'Apply APS to Scene',
            'Are you sure you want to apply APS to the scene?')
        if msg == QtGui.QMessageBox.StandardButton.Yes:
            self.model.getScene().aps.apply_model()

    def removeAPS(self):
        self.model.getScene().aps.remove_model()


class KiteAPSPlot(KitePlot):

    class TopoPatchROI(pg.RectROI):
        def _makePen(self):
            # Generate the pen color for this ROI based on its current state.
            if self.mouseHovering:
                return pen_roi_highlight
            else:
                return self.pen

    def __init__(self, model):
        self.components_available = {
            'displacement':
            ['Displacement', lambda sp: sp.scene.displacement]
        }
        self._component = 'displacement'

        KitePlot.__init__(
            self, model=model, los_arrow=False, auto_downsample=True)

        llE, llN, sizeE, sizeN = self.model.aps.get_patch_coords()
        self.roi = self.TopoPatchROI(
            pos=(llE, llN),
            size=(sizeE, sizeN),
            sideScalers=True,
            pen=pen_roi)

        self.roi.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.roi.sigRegionChangeFinished.connect(self.updateTopoRegion)

        self.addItem(self.roi)

    @QtCore.pyqtSlot()
    def updateTopoRegion(self):
        # data = self.roi.getArrayRegion(self.image.image, self.image)
        # data[data == 0.] = num.nan
        # if num.all(num.isnan(data)):
        #     return

        llE, llN = self.roi.pos()
        sizeE, sizeN = self.roi.size()
        patch_coords = (llE, llN, sizeE, sizeN)
        self.model.aps.set_patch_coords(*patch_coords)

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
            antialias=True,
            brush=brush_aps,
            pen=pen_aps,
            size=4)

        self.aps_model = pg.PlotDataItem(
            antialias=True,
            pen=pen_aps_model)

        self.legend = pg.LegendItem(offset=(0., .5))

        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.aps_model, '')

        self.addItem(self.aps_correlation)
        self.addItem(self.aps_model)

        self.plot.setLabels(
            bottom='Elevation (m)',
            left='Displacement (m)')

    @QtCore.pyqtSlot()
    def update(self):
        aps = self.model.aps
        elevation, displacement = aps.get_data()

        step = max(1, displacement.size // self.MAXPOINTS)

        self.aps_correlation.setData(
            elevation[::step],
            displacement[::step])

        slope, intercept = aps.get_correlation()
        elev = num.array([elevation.min(), elevation.max()])
        model = elev * slope + intercept

        self.aps_model.setData(elev, model)

        self.legend.items[-1][1].setText('Slope %.4f m / km' % (slope * km))

    def activatePlot(self):
        self.model.sigSceneChanged.connect(self.update)
        self.model.sigAPSChanged.connect(self.update)
        self.update()

    def deactivatePlot(self):
        self.model.sigSceneChanged.disconnect(self.update)
        self.model.sigAPSChanged.disconnect(self.update)
