#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num

from PySide import QtGui, QtCore
import pyqtgraph as pg
from .tab import QKiteDock, QKitePlot
from .tab_scene import QKiteSceneParamMeta, QKiteSceneParamFrame
from ..covariance import modelCovariance

analy_pen0 = pg.mkPen((51, 53, 119, 0), width=1.5)
analy_pen1 = pg.mkPen((170, 57, 57, 255), width=2.5, style=QtCore.Qt.DotLine)
analy_pen2 = pg.mkPen((45, 136, 45, 200), width=2.5, style=QtCore.Qt.DashLine)


class QKiteCovarianceDock(QKiteDock):
    def __init__(self, covariance):
        self.title = 'Scene.displacement'
        self.main_widget = QKiteNoisePlot(covariance)
        self.tools = {
            'Covariance.noiseSpectrum':
                QKiteNoisePowerspec(self.main_widget),
            'Covariance.covariance_func':
                QKiteCovariogram(self.main_widget),
            'Covariance.structure_func':
                QKiteStructureFunction(self.main_widget),
        }

        self.parameters = [
            QKiteSceneParamFrame(covariance._scene, expanded=False),
            QKiteSceneParamMeta(covariance._scene, expanded=False),
        ]

        QKiteDock.__init__(self, covariance)

        for dock in self.tool_docks:
            dock.setStretch(10, .5)


class QKiteNoisePlot(QKitePlot):
    def __init__(self, covariance):
        self.components_available = {
            'displacement': ['Displacement',
                             lambda cov: cov._scene.displacement],
        }

        self._component = 'displacement'
        roi_pen = pg.mkPen((45, 136, 45), width=3)

        QKitePlot.__init__(self, container=covariance)
        self.covariance = self.container

        self.roi = pg.RectROI(*self.covariance.noise_coord,
                              sideScalers=True,
                              pen=roi_pen)
        self.roi.sigRegionChangeFinished.connect(self.updateNoiseRegion)
        self.addItem(self.roi)

    def updateNoiseRegion(self):
        data = self.roi.getArrayRegion(self.image.image, self.image)
        data[data == 0.] = num.nan
        self.covariance.noise_data = data


class _QKiteCovariancePlot(QtGui.QWidget):
    def __init__(self, parent_plot):
        QtGui.QWidget.__init__(self)
        self.parent_plot = parent_plot
        self.covariance = parent_plot.covariance

        self.plot = pg.PlotWidget(background='default')
        self.plot.showGrid(True, True, alpha=.5)
        self.plot.setMenuEnabled(False)
        self.plot.enableAutoRange()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plot)

    def addItem(self, *args, **kwargs):
        self.plot.addItem(*args, **kwargs)


class QKiteNoisePowerspec(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)

        self.power_plot = pg.PlotDataItem(antialias=True)
        self.power_analy0 = pg.PlotDataItem(antialias=True, pen=analy_pen0)
        self.power_analy1 = pg.PlotDataItem(antialias=True, pen=analy_pen1)
        self.power_analy2 = pg.PlotDataItem(antialias=True, pen=analy_pen2)

        self.power_plot.setZValue(10)
        self.plot.setLabels(bottom='Wavenumber (cycles/m)',
                            left='Power (m<sup>2</sup>)')
        self.plot.setLogMode(x=True, y=True)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.power_analy2, 'Log Linear Model')

        self.addItem(self.power_plot)
        self.addItem(self.power_analy0)
        self.addItem(self.power_analy1)
        self.addItem(self.power_analy2)

        self.covariance.covarianceUpdate.subscribe(self.updatePowerPlot)
        self.updatePowerPlot()

    def updatePowerPlot(self):
        spec, k, _, _, _ = self.covariance.noiseSpectrum()
        self.power_plot.setData(k, spec)
        # self.power_analy0.setData(
        #     k, self.plot.covariance.powerspecAnalytical(k, 0))
        # self.power_analy1.setData(
        #     k, self.plot.covariance.powerspecAnalytical(k, 1))
        self.power_analy2.setData(
            k, self.covariance.powerspecAnalytical(k, 3))


class QKiteCovariogram(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.covariogram = pg.PlotDataItem(antialias=True)
        self.covariogram.setZValue(10)
        self.cov_analytical0 = pg.PlotDataItem(antialias=True, pen=analy_pen0)
        self.cov_analytical1 = pg.PlotDataItem(antialias=True, pen=analy_pen1)
        self.cov_analytical2 = pg.PlotDataItem(antialias=True, pen=analy_pen2)

        self.addItem(self.covariogram)
        self.addItem(self.cov_analytical0)
        self.addItem(self.cov_analytical1)
        self.addItem(self.cov_analytical2)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_analytical1, '')
        self.legend.template = 'Model: {0:.5f} e^(-d/{1:.1f})'

        self.covariance.covarianceUpdate.subscribe(
            self.updateCovariancePlot)

        self.updateCovariancePlot()

    def updateCovariancePlot(self):
        cov, dist = self.covariance.covariance_func

        self.covariogram.setData(dist, cov)
        self.cov_analytical0.setData(
            dist, self.covariance.covarianceAnalytical(0)[0])
        self.cov_analytical1.setData(
            dist,
            modelCovariance(dist,
                            *self.covariance.covarianceModelFit(3)))
        self.cov_analytical2.setData(
            dist, self.covariance.covarianceAnalytical(3)[0])

        self.legend.items[-1][1].setText(
            self.legend.template.format(
                *self.covariance.covarianceModelFit(3)))


class QKiteStructureFunction(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)

        self.structure = pg.PlotDataItem(antialias=True)
        self.variance = pg.InfiniteLine(
            pen=analy_pen1,
            angle=0, movable=True, hoverPen=None,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.))})
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.addItem(self.structure)
        self.addItem(self.variance)
        self.covariance.covarianceUpdate.subscribe(
            self.updateStructurePlot)
        self.variance.sigPositionChanged.connect(
            self.changeVariance)

        self.updateStructurePlot()

    def updateStructurePlot(self):
        struc, dist = self.covariance.structure_func
        self.structure.setData(dist, struc)
        self.variance.setValue(self.covariance.variance)

    def changeVariance(self, inf_line):
        self.covariance.variance = inf_line.getYPos()
