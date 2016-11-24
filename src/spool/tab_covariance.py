#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num

from PySide import QtGui, QtCore
import pyqtgraph as pg
from .tab import QKiteDock, QKitePlot
from ..covariance import modelCovariance

plot_padding = .1

analy_pen0 = pg.mkPen((51, 53, 119, 0), width=1.5)
analy_pen1 = pg.mkPen((170, 57, 57, 200), width=1.5, style=QtCore.Qt.DotLine)
analy_pen2 = pg.mkPen((45, 136, 45), width=2.5, style=QtCore.Qt.DashLine)


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
        # self.power_analy0.setData(
        #     k, self.plot.covariance.powerspecAnalytical(k, 0))
        # self.power_analy1.setData(
        #     k, self.plot.covariance.powerspecAnalytical(k, 1))
        self.power_analy2.setData(
            k, self.plot.covariance.powerspecAnalytical(k, 3))


class QKiteCovariogram(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        self.covariogram = pg.PlotDataItem(antialias=True)
        self.covariogram.setZValue(10)
        self.cov_analytical0 = pg.PlotDataItem(antialias=True, pen=analy_pen0)
        self.cov_analytical1 = pg.PlotDataItem(antialias=True, pen=analy_pen1,
                                               name='Exp. Fit - Linear Power')
        self.cov_analytical2 = pg.PlotDataItem(antialias=True, pen=analy_pen2,
                                               name='Linear Power')

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

        self.legend = pg.LegendItem(offset=(.5, .5))
        self.legend.setParentItem(self.plt_wdgt.graphicsItem())
        self.legend.addItem(self.cov_analytical1, 'Exp. Fit - Linear Power')
        self.legend.addItem(self.cov_analytical2, 'Linear Power')

        self.plot.covariance.covarianceUpdate.subscribe(
            self.updateCovariancePlot)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plt_wdgt)

        self.updateCovariancePlot()

    def updateCovariancePlot(self):
        cov, dist = self.plot.covariance.covariance_func

        self.covariogram.setData(dist, cov)
        self.cov_analytical0.setData(
            dist, self.plot.covariance.covarianceAnalytical(0)[0])
        self.cov_analytical1.setData(
            dist,
            modelCovariance(dist,
                            *self.plot.covariance.covarianceModelFit(3)))
        self.cov_analytical2.setData(
            dist, self.plot.covariance.covarianceAnalytical(3)[0])

        padx = (dist.max()-dist.min()) * plot_padding
        pady = (cov.max()-cov.min()) * plot_padding
        self.plt_wdgt.setLimits(xMin=dist.min()-padx,
                                xMax=dist.max()+padx,
                                yMin=cov.min()-pady,
                                yMax=cov.max()+pady)


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

        padx = (dist.max()-dist.min()) * plot_padding
        pady = (struc.max()-struc.min()) * plot_padding
        self.plt_wdgt.setLimits(xMin=dist.min()-padx,
                                xMax=dist.max()+padx,
                                yMin=struc.min()-pady,
                                yMax=struc.max()+pady)
