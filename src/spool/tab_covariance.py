#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num

from PySide import QtGui
import pyqtgraph as pg  # noqa
import numpy as num  # noqa
from .tab import QKiteDock, QKitePlot


class QKiteCovarianceDock(QKiteDock):
    def __init__(self, covariance):
        self.title = 'Scene Covariance'
        self.main_widget = QKiteNoisePlot
        self.tools = {
            'Powerspectrum': QKiteNoisePowerspec,
            'Covariogram': QKiteCovariogram,
            'Structure Function': QKiteStructureFunction,
        }

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


class QKiteNoisePowerspec(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        analy_pen0 = pg.mkPen((170, 57, 57), width=2.4)
        analy_pen1 = pg.mkPen((51, 53, 119), width=2.4)
        analy_pen2 = pg.mkPen((45, 136, 45), width=2.4)

        self.power_plot = pg.PlotDataItem(antialias=True)
        self.power_analy0 = pg.PlotDataItem(antialias=True, pen=analy_pen0)
        self.power_analy1 = pg.PlotDataItem(antialias=True, pen=analy_pen1)
        self.power_analy2 = pg.PlotDataItem(antialias=True, pen=analy_pen2)

        self.power_plot.setZValue(10)
        self.plt_wdgt = pg.PlotWidget(background='default')
        self.plt_wdgt.setLabels(bottom='Wavenumber (cycles/m)',
                                left='Power (m<sup>2</sup>)')
        self.plt_wdgt.setLogMode(x=True, y=True)
        self.plt_wdgt.setMenuEnabled(False)
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.enableAutoRange()

        self.plt_wdgt.addItem(self.power_plot)

        self.plt_wdgt.addItem(self.power_analy0)
        self.plt_wdgt.addItem(self.power_analy1)
        self.plt_wdgt.addItem(self.power_analy2)

        self.plot.covariance.covarianceUpdate.subscribe(self.updatePowerPlot)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plt_wdgt)

        self.updatePowerPlot()

    def updatePowerPlot(self):
        spec, k, _, _, _ = self.plot.covariance.noiseSpectrum()
        self.power_plot.setData(k, spec)
        self.power_analy0.setData(
            k, self.plot.covariance.powerspecAnalytical(k, 0))
        self.power_analy1.setData(
            k, self.plot.covariance.powerspecAnalytical(k, 1))
        self.power_analy2.setData(
            k, self.plot.covariance.powerspecAnalytical(k, 2))
        # self.plt_wdgt.setLimits(xMin=k.min(), xMax=k.max(),
        #                         yMin=spec.min(), yMax=spec.max())


class QKiteCovariogram(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot
        analy_pen0 = pg.mkPen((170, 57, 57), width=2.4)
        analy_pen1 = pg.mkPen((51, 53, 119), width=2.4)
        analy_pen2 = pg.mkPen((45, 136, 45), width=2.4)

        self.covariogram = pg.PlotDataItem(antialias=True)
        self.covariogram.setZValue(10)
        self.cov_analytical0 = pg.PlotDataItem(antialias=True, pen=analy_pen0)
        self.cov_analytical1 = pg.PlotDataItem(antialias=True, pen=analy_pen1)
        self.cov_analytical2 = pg.PlotDataItem(antialias=True, pen=analy_pen2)

        self.plt_wdgt = pg.PlotWidget(background='default')
        self.plt_wdgt.setLabels(bottom={'Distance', 'm'},
                                left='Covariance (m<sup>2</sup>)')
        self.plt_wdgt.setMenuEnabled(False)
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.enableAutoRange()

        self.plt_wdgt.addItem(self.covariogram)
        self.plt_wdgt.addItem(self.cov_analytical0)
        self.plt_wdgt.addItem(self.cov_analytical1)
        self.plt_wdgt.addItem(self.cov_analytical2)

        self.plot.covariance.covarianceUpdate.subscribe(
            self.updateCovariancePlot)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plt_wdgt)

        self.updateCovariancePlot()

    def updateCovariancePlot(self):
        # padding = 20

        cov, dist = self.plot.covariance.covariance_func
        self.covariogram.setData(dist, cov)
        self.cov_analytical0.setData(
            dist, self.plot.covariance.covarianceAnalytical(0)[0])
        self.cov_analytical1.setData(
            dist, self.plot.covariance.covarianceAnalytical(1)[0])
        self.cov_analytical2.setData(
            dist, self.plot.covariance.covarianceAnalytical(2)[0])
#        self.plt_wdgt.setLimits(xMin=dist.min()+padding,
#                                xMax=dist.max()+padding,
#                                yMin=cov.min()+padding,
#                                yMax=cov.max()+padding)


class QKiteStructureFunction(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot
        self.structure = pg.PlotDataItem(antialias=True)

        self.plt_wdgt = pg.PlotWidget(background='default')
        self.plt_wdgt.setLabels(bottom={'Distance', 'm'},
                                left='Covariance (m<sup>2</sup>)')
        self.plt_wdgt.setMenuEnabled(False)
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.enableAutoRange()

        self.plt_wdgt.addItem(self.structure)
        self.plot.covariance.covarianceUpdate.subscribe(
            self.updateStructurePlot)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plt_wdgt)

        self.updateStructurePlot()

    def updateStructurePlot(self):
        struc, dist = self.plot.covariance.structure_func
        self.structure.setData(dist, struc)
