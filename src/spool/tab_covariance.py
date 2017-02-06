#!/bin/python
from __future__ import division, absolute_import, print_function, \
    unicode_literals
import numpy as num
import pyqtgraph as pg
from pyqtgraph import dockarea

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
pen_variance = pg.mkPen((94, 151, 50, 200), width=2.5,
                        style=QtCore.Qt.DashLine)
pen_green_dash = pg.mkPen((45, 136, 45, 200), width=2.5,
                          style=QtCore.Qt.DashLine)


class QKiteCovariance(QKiteView):
    title = 'Scene.covariance'

    def __init__(self, spool):
        scene_proxy = spool.scene_proxy

        covariance_plot = QKiteNoisePlot(scene_proxy)
        self.main_widget = covariance_plot
        self.tools = {
            'Covariance.powerspecNoise':
                QKiteNoisePowerspec(covariance_plot),
            'Covariance.covariance_func':
                QKiteCovariogram(covariance_plot),
            # 'Covariance.structure_func':
            #     QKiteStructureFunction(covariance_plot),
        }

        self.param_covariance = QKiteParamCovariance(scene_proxy)
        self.parameters = [self.param_covariance]

        self.dialogCovarianceNoise = QKiteToolNoiseData(scene_proxy, spool)
        self.dialogCovarianceMatrix = QKiteToolWeightMatrix(scene_proxy, spool)
        self.dialogCovarianceNoiseSyn =\
            QKiteToolSyntheticNoise(scene_proxy, spool)
        self.dialogCovarianceNoise.showSynthetic.released.connect(
            self.dialogCovarianceNoiseSyn.show)

        spool.actionCovariance_Noise.triggered.connect(
            self.dialogCovarianceNoise.show)
        covariance_plot.roi.sigClicked.connect(
            self.dialogCovarianceNoise.show)
        spool.actionCovariance_Matrix.triggered.connect(
            self.dialogCovarianceMatrix.show)
        spool.actionCovariance_Noise_Synthetic.triggered.connect(
            self.dialogCovarianceNoiseSyn.show)

        spool.actionCovariance_Noise.setEnabled(True)
        spool.actionCovariance_Matrix.setEnabled(True)
        spool.actionCovariance_Noise_Synthetic.setEnabled(True)

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


class _QKiteSubplotPlot(QtGui.QWidget):
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


class QKiteNoisePowerspec(_QKiteSubplotPlot):
    def __init__(self, parent_plot):
        _QKiteSubplotPlot.__init__(self, parent_plot)

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
        # self.addItem(self.power_lin)

        self.scene_proxy.sigCovarianceChanged.connect(self.update)
        self.update()

    def update(self):
        covariance = self.scene_proxy.covariance
        spec, k, _, _, _, _ = covariance.powerspecNoise1D()
        self.power.setData(k, spec)
        self.power_lin.setData(
            k, covariance.powerspecModel(k))


class QKiteCovariogram(_QKiteSubplotPlot):
    def __init__(self, parent_plot):
        _QKiteSubplotPlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.cov = pg.PlotDataItem(antialias=True)
        self.cov.setZValue(10)
        self.cov_model = pg.PlotDataItem(antialias=True, pen=pen_red_dot)
        self.cov_lin_pow = pg.PlotDataItem(antialias=True, pen=pen_green_dash)
        self.variance = pg.InfiniteLine(
            pen=pen_variance,
            angle=0, movable=True, hoverPen=None,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.)),
                       'color': pg.mkColor(255, 255, 255, 155)})
        self.addItem(self.variance)

        self.addItem(self.cov)
        self.addItem(self.cov_model)
        # self.addItem(self.cov_lin_pow)

        self.legend = pg.LegendItem(offset=(0., .5))

        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_model, '')
        self.legend.template = 'Model: {0:.5f} e^(-d/{1:.1f}) | RMS: {rms:.4e}'

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

        self.legend.items[-1][1].setText(
            self.legend.template.format(
                *covariance.covariance_model,
                rms=covariance.covariance_model_rms))
        self.variance.setValue(covariance.variance)


class QKiteStructureFunction(_QKiteSubplotPlot):
    def __init__(self, parent_plot):
        _QKiteSubplotPlot.__init__(self, parent_plot)

        self.structure = pg.PlotDataItem(antialias=True)
        self.variance = pg.InfiniteLine(
            pen=pen_red_dot,
            angle=0, movable=True, hoverPen=None,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.)),
                       'color': pg.mkColor(255, 255, 255, 155)})
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


class QKiteToolNoiseData(QtGui.QDialog):
    class noise_plot(QKitePlot):
        def __init__(self, scene_proxy):
            self.components_available = {
                'noise_data': [
                  'Displacement',
                  lambda sp: self.noise_data_masked(sp.covariance)
                ]}

            self._component = 'noise_data'
            QKitePlot.__init__(self, scene_proxy=scene_proxy)

        @staticmethod
        def noise_data_masked(covariance):
            data = covariance.noise_data.copy()
            data[data == 0.] = num.nan
            return data

        def proxy_connect(self):
            self.scene_proxy.sigCovarianceChanged.connect(self.update)

        def proxy_disconnect(self):
            self.scene_proxy.sigCovarianceChanged.disconnect(self.update)

    def __init__(self, scene_proxy, parent=None):
        QtGui.QDialog.__init__(self, parent)

        cov_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/covariance_noise.ui')
        loadUi(cov_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        self.noise_patch = self.noise_plot(scene_proxy)
        noise_colormap = QKiteToolColormap(self.noise_patch)

        self.dockarea = dockarea.DockArea(self)
        self.dockarea.addDock(
            dockarea.Dock('Covariance.noise_data',
                          widget=self.noise_patch,
                          size=(4, 4),
                          autoOrientation=False,),
            position='left')
        self.dockarea.addDock(
            dockarea.Dock('Colormap',
                          widget=noise_colormap,
                          size=(1, 1),
                          autoOrientation=False,),
            position='right')
        self.horizontalLayoutPlot.addWidget(self.dockarea)

    def closeEvent(self, ev):
        self.noise_patch.proxy_disconnect()
        ev.accept()

    def showEvent(self, ev):
        self.noise_patch.update()
        self.noise_patch.proxy_connect()
        ev.accept()


class QKiteToolWeightMatrix(QtGui.QDialog):
    class weight_plot(QKitePlot):
        def __init__(self, scene_proxy):
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            self._component = 'weight'

            QKitePlot.__init__(self, scene_proxy)
            self.scene_proxy = scene_proxy

            gradient = Gradients['thermal']

            self.cmap = pg.ColorMap(pos=[c[0] for c in gradient['ticks']],
                                    color=[c[1] for c in gradient['ticks']],
                                    mode=gradient['mode'])
            self.image.setLookupTable(self.cmap.getLookupTable())

            self.setLabels(bottom={'Leaf #', ''},
                           left={'Leaf #', ''})
            self.setAspectLocked(True)
            self.setMouseEnabled(x=False, y=False)
            self.hint = {
                'leaf1': 0,
                'leaf2': 0,
                'weight': num.nan,
            }

            self.hint_text.template =\
                '<span style="font-family: monospace; color: #fff;'\
                'background-color: #000;">'\
                'Leaf #1: {leaf1:d} | Leaf #2: {leaf2:d} | '\
                'Weight {weight:e}</span>'

            self.update()

        def update(self):
            self.image.updateImage(
                self.scene_proxy.covariance.weight_matrix_focal.T,
                autoLevels=True)

        def transFromFrame(self):
            # self.resetTransform()
            self.setRange(xRange=(0, self.scene_proxy.quadtree.nleafs),
                          yRange=(0, self.scene_proxy.quadtree.nleafs))

        @QtCore.Slot(object)
        def mouseMoved(self, event=None):
            if event is None:
                pass
            elif self.image.sceneBoundingRect().contains(event[0]):
                map_pos = self.plotItem.vb.mapSceneToView(event[0])
                if not map_pos.isNull():
                    img_pos = self.image.mapFromScene(event).data
                    value = self.image.image[int(img_pos().x()),
                                             int(img_pos().y())]

                    self.hint['leaf1'] = int(map_pos.x())
                    self.hint['leaf2'] = int(map_pos.y())
                    self.hint['weight'] = value
            self.hint_text.setText(self.hint_text.template.format(**self.hint))

        def proxy_connect(self):
            self.scene_proxy.sigCovarianceChanged.connect(self.update)
            self.scene_proxy.sigQuadtreeChanged.connect(self.transFromFrame)

        def proxy_disconnect(self):
            self.scene_proxy.sigCovarianceChanged.disconnect(self.update)
            self.scene_proxy.sigQuadtreeChanged.disconnect(self.transFromFrame)

    def __init__(self, scene_proxy, parent=None):
        QtGui.QDialog.__init__(self, parent)

        cov_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/covariance_matrix.ui')
        loadUi(cov_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        self.weight_matrix = self.weight_plot(scene_proxy)
        self.dockarea = dockarea.DockArea(self)
        self.dockarea.addDock(
            dockarea.Dock('Covariance.weight_matrix_focal',
                          widget=self.weight_matrix,
                          size=(4, 4),
                          autoOrientation=False,),
            position='left')
        self.horizontalLayoutPlot.addWidget(self.dockarea)

    def closeEvent(self, ev):
        self.weight_matrix.proxy_disconnect()
        ev.accept()

    def showEvent(self, ev):
        self.weight_matrix.update()
        self.weight_matrix.proxy_connect()
        ev.accept()


class QKiteToolSyntheticNoise(QtGui.QDialog):
    class noise_plot(QKitePlot):
        def __init__(self, scene_proxy):
            self.components_available = {
                'synthetic_noise': [
                  'Noise',
                  lambda sp: sp.covariance.syntheticNoise(
                    sp.covariance.noise_data.shape)
                ]}

            self._component = 'synthetic_noise'
            QKitePlot.__init__(self, scene_proxy=scene_proxy)

        def proxy_connect(self):
            self.scene_proxy.sigCovarianceChanged.connect(self.update)

        def proxy_disconnect(self):
            self.scene_proxy.sigCovarianceChanged.disconnect(self.update)

    def __init__(self, scene_proxy, parent=None):
        QtGui.QDialog.__init__(self, parent)

        cov_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/covariance_noise-synthetic.ui')
        loadUi(cov_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))
        self.generateNoise.setIcon(self.style().standardPixmap(
                                   QtGui.QStyle.SP_BrowserReload))

        self.noise_syn = self.noise_plot(scene_proxy)
        self.params = self.getParameterTree()
        noise_colormap = QKiteToolColormap(self.noise_syn)

        self.generateNoise.released.connect(self.noise_syn.update)

        self.dockarea = dockarea.DockArea(self)
        self.dockarea.addDock(
            dockarea.Dock('Covariance.syntheticNoise',
                          widget=self.noise_syn,
                          size=(4, 4),
                          autoOrientation=False,),
            position='left')
        self.dockarea.addDock(
            dockarea.Dock('Colormap',
                          widget=noise_colormap,
                          size=(1, 1),
                          autoOrientation=False,),
            position='right')
        self.dockarea.addDock(
            dockarea.Dock('Synthetic Noise Parameters',
                          widget=self.params,
                          size=(1, 1),
                          autoOrientation=False,),
            position='left')
        self.horizontalLayoutPlot.addWidget(self.dockarea)

    def getParameterTree(self):
        pt = pg.parametertree
        pTypes = pt.parameterTypes
        tree = pt.ParameterTree()
        p = pTypes.WidgetParameterItem(name='Width', type='int')
        tree.addChild(p)
        return tree

    def update_params(self):
        self.noise_syn.components_available = {
            'synthetic_noise': [
              'Noise',
              lambda sp: sp.covariance.syntheticNoise(
                sp.covariance.noise_data.shape)
            ]}

    def closeEvent(self, ev):
        self.noise_syn.proxy_disconnect()
        ev.accept()

    def showEvent(self, ev):
        self.noise_syn.update()
        self.noise_syn.proxy_connect()
        ev.accept()


class QKiteParamCovariance(QKiteParameterGroup):
    def __init__(self, scene_proxy, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.covariance'

        self.parameters = OrderedDict([
            ('powerspec_model_rms', None),
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
