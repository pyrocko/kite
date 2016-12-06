#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num
import pyqtgraph as pg

from os import path
from PySide import QtGui, QtCore
from .common import (QKiteView, QKitePlot, QKiteParameterGroup,
                     QKiteToolColormap)
from .utils_qt import loadUi
from ..covariance import modelCovariance

analy_pen0 = pg.mkPen((51, 53, 119, 0), width=1.5)
pen_red_dot = pg.mkPen((170, 57, 57, 255), width=2.5,
                       style=QtCore.Qt.DotLine)
pen_green_dash = pg.mkPen((45, 136, 45, 200), width=2.5,
                          style=QtCore.Qt.DashLine)


class QKiteCovariance(QKiteView):
    def __init__(self, spool):
        covariance = spool.scene.covariance
        self.title = 'Scene.covariance'
        self.main_widget = QKiteNoisePlot(covariance)
        self.tools = {
            'Covariance.noiseSpectrum':
                QKiteNoisePowerspec(self.main_widget),
            'Covariance.covariance_func':
                QKiteCovariogram(self.main_widget),
            'Covariance.structure_func':
                QKiteStructureFunction(self.main_widget),
        }

        self.parameters = [QKiteParamCovariance(spool, expanded=False)]

        self.dialogCovariance = QKiteToolCovariance(covariance, spool)
        spool.actionCovariance.triggered.connect(self.dialogCovariance.show)
        spool.actionCovariance.setEnabled(True)

        QKiteView.__init__(self)

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

        llE, llN, sizeE, sizeN = self.covariance.noise_coord
        self.roi = pg.RectROI((llE, llN), (sizeE, sizeN),
                              sideScalers=True,
                              pen=roi_pen)
        self.roi.sigRegionChangeFinished.connect(self.updateNoiseRegion)
        self.addItem(self.roi)

    def updateNoiseRegion(self):
        data = self.roi.getArrayRegion(self.image.image, self.image)
        data[data == 0.] = num.nan
        if num.all(num.isnan(data)):
            return

        llE, llN = self.roi.pos()
        sizeE, sizeN = self.roi.size()
        self.covariance.noise_coord = (llE, llN, sizeE, sizeN)
        self.covariance.noise_data = data.T


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

        self.power = pg.PlotDataItem(antialias=True)
        self.power_lin = pg.PlotDataItem(antialias=True, pen=pen_green_dash)

        self.power.setZValue(10)
        self.plot.setLabels(bottom='Wavenumber (cycles/m)',
                            left='Power (m<sup>2</sup>)')
        self.plot.setLogMode(x=True, y=True)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.power_lin, 'Log-linear model')

        self.addItem(self.power)
        self.addItem(self.power_lin)

        self.covariance.evCovarianceUpdate.subscribe(self.updatePowerPlot)
        self.updatePowerPlot()

    def updatePowerPlot(self):
        spec, k, _, _, _ = self.covariance.noiseSpectrum()
        self.power.setData(k, spec)
        self.power_lin.setData(k, self.covariance.powerspecAnalytical(k, 3))


class QKiteCovariogram(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.cov = pg.PlotDataItem(antialias=True)
        self.cov.setZValue(10)
        self.cov_model = pg.PlotDataItem(antialias=True, pen=pen_red_dot)
        self.cov_lin_pow = pg.PlotDataItem(antialias=True, pen=pen_green_dash)
        self.misfit_label = pg.LabelItem(text='', justify='right', size='8pt',
                                         parent=self.plot.plotItem)
        self.misfit_label.anchor(itemPos=(0., 0.), parentPos=(.2, .1))
        self.misfit_label.format = 'Misfit: {0:.6f}'

        self.addItem(self.cov)
        self.addItem(self.cov_model)
        self.addItem(self.cov_lin_pow)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_model, '')
        self.legend.template = 'Model: {0:.5f} e^(-d/{1:.1f})'

        self.covariance.evCovarianceUpdate.subscribe(
            self.updateCovariancePlot)

        self.updateCovariancePlot()

    def updateCovariancePlot(self):
        cov, dist = self.covariance.covariance_func

        self.cov.setData(dist, cov)
        self.cov_model.setData(
            dist, modelCovariance(dist, *self.covariance.covariance_model))
        self.cov_lin_pow.setData(
            dist, self.covariance.covarianceAnalytical(3)[0])
        self.misfit_label.setText(
            self.misfit_label.format.format(
                    self.covariance.covariance_model_misfit))

        self.legend.items[-1][1].setText(
            self.legend.template.format(
                *self.covariance.covariance_model))


class QKiteStructureFunction(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)

        self.structure = pg.PlotDataItem(antialias=True)
        self.variance = pg.InfiniteLine(
            pen=pen_red_dot,
            angle=0, movable=True, hoverPen=None,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.))})
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.addItem(self.structure)
        self.addItem(self.variance)
        self.covariance.evCovarianceUpdate.subscribe(
            self.updateStructurePlot)
        self.variance.sigPositionChangeFinished.connect(
            self.changeVariance)

        self.updateStructurePlot()

    def updateStructurePlot(self):
        struc, dist = self.covariance.structure_func
        self.structure.setData(dist, struc)
        self.variance.setValue(self.covariance.variance)

    def changeVariance(self, inf_line):
        self.covariance.variance = inf_line.getYPos()


class QKiteToolCovariance(QtGui.QDialog):
    class noise_plot(QKitePlot):
        def __init__(self, covariance):
            self.components_available = {
                'noise_data': [
                    'Displacement',
                    lambda cov: self.noise_data_masked(cov)
                    ]}

            self._component = 'noise_data'
            QKitePlot.__init__(self, container=covariance)
            self.covariance = self.container
            self.covariance.evCovarianceUpdate.subscribe(self.update)

        @staticmethod
        def noise_data_masked(covariance):
            data = covariance.noise_data.copy()
            data[data == 0.] = num.nan
            return data

    def __init__(self, covariance, parent=None):
        QtGui.QDialog.__init__(self, parent)

        cov_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/covariance.ui')
        loadUi(cov_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        noise_patch = self.noise_plot(covariance)
        noise_colormap = QKiteToolColormap(noise_patch)
        self.horizontalLayoutPlot.addWidget(noise_patch)
        self.horizontalLayoutPlot.addWidget(noise_colormap)


class QKiteParamCovariance(QKiteParameterGroup):
    def __init__(self, spool, **kwargs):
        covariance = spool.scene.covariance
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.covariance'

        self.parameters = {'variance': None,
                           'covariance_model [a]':
                           lambda c: c.covariance_model[0],
                           'covariance_model [b]':
                           lambda c: c.covariance_model[1],
                           'covariance_model_misfit': None,
                           'noise_patch_size_km2': None}
        QKiteParameterGroup.__init__(self, covariance, **kwargs)
        covariance.evCovarianceUpdate.subscribe(self.updateValues)
