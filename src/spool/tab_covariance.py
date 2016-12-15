#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num
import pyqtgraph as pg

from os import path
from PySide import QtGui, QtCore
from collections import OrderedDict
from .utils_qt import loadUi
from ..covariance import modelCovariance
from .common import (QKiteView, QKitePlot, QKiteParameterGroup,
                     QKiteToolColormap)

analy_pen0 = pg.mkPen((51, 53, 119, 0), width=1.5)
pen_red_dot = pg.mkPen((170, 57, 57, 255), width=2.5,
                       style=QtCore.Qt.DotLine)
pen_green_dash = pg.mkPen((45, 136, 45, 200), width=2.5,
                          style=QtCore.Qt.DashLine)


class QKiteCovariance(QKiteView):
    def __init__(self, spool):
        self.title = 'Scene.covariance'
        scene_proxy = spool.scene_proxy

        covariance_plot = QKiteNoisePlot(scene_proxy)
        self.main_widget = covariance_plot
        self.tools = {
            'Covariance.noiseSpectrum':
                QKiteNoisePowerspec(covariance_plot),
            'Covariance.covariance_func':
                QKiteCovariogram(covariance_plot),
            'Covariance.structure_func':
                QKiteStructureFunction(covariance_plot),
        }

        self.param_covariance = QKiteParamCovariance(scene_proxy)
        self.parameters = [self.param_covariance]

        self.dialogCovariance = QKiteToolCovariance(scene_proxy, spool)
        spool.actionCovariance.triggered.connect(self.dialogCovariance.show)
        covariance_plot.roi.sigClicked.connect(self.dialogCovariance.show)
        spool.actionCovariance.setEnabled(True)

        scene_proxy.sigSceneModelChanged.connect(self.modelChanged)

        QKiteView.__init__(self)

        for dock in self.tool_docks:
            dock.setStretch(10, .5)

    def modelChanged(self):
        self.dialogCovariance.close()
        self.main_widget.onConfigChanged()

        self.param_covariance.updateValues()
        for v in self.tools.itervalues():
            v.update()


class QKiteNoisePlot(QKitePlot):
    def __init__(self, scene_proxy):
        self.components_available = {
            'displacement':
            ['Displacement', lambda sp: sp.scene.displacement],
        }
        self._component = 'displacement'

        QKitePlot.__init__(self, scene_proxy=scene_proxy)

        llE, llN, sizeE, sizeN = self.scene_proxy.covariance.noise_coord
        roi_pen = pg.mkPen((45, 136, 45), width=3)
        self.roi = pg.RectROI((llE, llN), (sizeE, sizeN),
                              sideScalers=True,
                              pen=roi_pen)
        self.roi.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.roi.sigRegionChangeFinished.connect(self.updateNoiseRegion)
        self.addItem(self.roi)

        self.scene_proxy.sigCovarianceConfigChanged.connect(
            self.onConfigChanged)

    def onConfigChanged(self):
        llE, llN, sizeE, sizeN = self.scene_proxy.covariance.noise_coord
        self.roi.setPos((llE, llN), update=False, finish=False)
        self.roi.setSize((sizeE, sizeN), finish=False)
        self.update()
        self.transFromFrame()

    def updateNoiseRegion(self):
        data = self.roi.getArrayRegion(self.image.image, self.image)
        data[data == 0.] = num.nan
        if num.all(num.isnan(data)):
            return

        llE, llN = self.roi.pos()
        sizeE, sizeN = self.roi.size()
        self.scene_proxy.covariance.noise_coord = (llE, llN, sizeE, sizeN)
        self.scene_proxy.covariance.noise_data = data.T


class _QKiteCovariancePlot(QtGui.QWidget):
    def __init__(self, parent_plot):
        QtGui.QWidget.__init__(self)
        self.parent_plot = parent_plot
        self.scene_proxy = parent_plot.scene_proxy

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

        self.scene_proxy.sigCovarianceChanged.connect(self.update)
        self.update()

    def update(self):
        covariance = self.scene_proxy.covariance
        spec, k, _, _, _ = covariance.noiseSpectrum()
        self.power.setData(k, spec)
        self.power_lin.setData(
            k, covariance.powerspecAnalytical(k, 3))


class QKiteCovariogram(_QKiteCovariancePlot):
    def __init__(self, parent_plot):
        _QKiteCovariancePlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.cov = pg.PlotDataItem(antialias=True)
        self.cov.setZValue(10)
        self.cov_model = pg.PlotDataItem(antialias=True, pen=pen_red_dot)
        self.cov_lin_pow = pg.PlotDataItem(antialias=True, pen=pen_green_dash)
        self.rms_label = pg.LabelItem(text='', justify='right', size='10pt',
                                      parent=self.plot.plotItem)
        self.rms_label.anchor(itemPos=(0., 0.), parentPos=(.15, .05))
        self.rms_label.format = 'RMS: {0:.4e}'

        self.addItem(self.cov)
        self.addItem(self.cov_model)
        self.addItem(self.cov_lin_pow)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_model, '')
        self.legend.template = 'Model: {0:.5f} e^(-d/{1:.1f})'

        self.scene_proxy.sigCovarianceChanged.connect(
            self.update)

        self.update()

    def update(self):
        covariance = self.scene_proxy.covariance
        cov, dist = covariance.covariance_func

        self.cov.setData(dist, cov)
        self.cov_model.setData(
            dist, modelCovariance(dist, *covariance.covariance_model))
        self.cov_lin_pow.setData(
            dist, covariance.covarianceAnalytical(3)[0])
        self.rms_label.setText(
            self.rms_label.format.format(
                    covariance.covariance_model_rms))

        self.legend.items[-1][1].setText(
            self.legend.template.format(
                *covariance.covariance_model))


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
        self.scene_proxy.sigCovarianceChanged.connect(
            self.update)
        self.variance.sigPositionChangeFinished.connect(
            self.changeVariance)

        self.update()

    def update(self):
        covariance = self.scene_proxy.covariance
        struc, dist = covariance.structure_func
        self.structure.setData(dist, struc)
        self.variance.setValue(covariance.variance)

    def changeVariance(self, inf_line):
        covariance = self.scene_proxy.covariance
        covariance.variance = inf_line.getYPos()


class QKiteToolCovariance(QtGui.QDialog):
    class noise_plot(QKitePlot):
        def __init__(self, scene_proxy):
            self.components_available = {
                'noise_data': [
                  'Displacement',
                  lambda sp: self.noise_data_masked(sp.covariance)
                ]}

            self._component = 'noise_data'
            QKitePlot.__init__(self, scene_proxy=scene_proxy)

            self.scene_proxy.sigCovarianceChanged.connect(self.update)

        @staticmethod
        def noise_data_masked(covariance):
            data = covariance.noise_data.copy()
            data[data == 0.] = num.nan
            return data

    def __init__(self, scene_proxy, parent=None):
        QtGui.QDialog.__init__(self, parent)

        cov_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/covariance.ui')
        loadUi(cov_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        noise_patch = self.noise_plot(scene_proxy)
        noise_colormap = QKiteToolColormap(noise_patch)

        self.horizontalLayoutPlot.addWidget(noise_patch)
        self.horizontalLayoutPlot.addWidget(noise_colormap)


class QKiteParamCovariance(QKiteParameterGroup):
    def __init__(self, scene_proxy, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.covariance'

        self.parameters = OrderedDict([
            ('variance', None),
            ('covariance_model [a]',
             lambda c: c.covariance_model[0]),
            ('covariance_model [b]',
             lambda c: c.covariance_model[1]),
            ('covariance_model_rms', None),
            ('noise_patch_size_km2', None),
            ('noise_patch_coord',
             lambda c: ', '.join([str(f) for f in c.noise_coord.tolist()])),
            ])

        scene_proxy.sigCovarianceChanged.connect(self.updateValues)
        QKiteParameterGroup.__init__(self,
                                     model=scene_proxy,
                                     model_attr='covariance',
                                     **kwargs)
