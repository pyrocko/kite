#!/bin/python
from __future__ import (division, absolute_import, print_function,
                        unicode_literals)
import numpy as num
import pyqtgraph as pg
from pyqtgraph import dockarea

from PySide import QtGui, QtCore
from collections import OrderedDict

from kite.qt_utils import loadUi
from kite.covariance import modelCovariance

from .common import (KiteView, KitePlot, KiteParameterGroup,
                     KiteToolColormap, get_resource)

analy_pen0 = pg.mkPen(
    (51, 53, 119, 0), width=1.5)
pen_red_dot = pg.mkPen(
    (170, 57, 57, 255), width=2.5, style=QtCore.Qt.DotLine)
pen_variance = pg.mkPen(
    (94, 151, 50, 200), width=2.5, style=QtCore.Qt.DashLine)
pen_green_dash = pg.mkPen(
    (45, 136, 45, 200), width=2.5, style=QtCore.Qt.DashLine)
pen_roi = pg.mkPen(
    (45, 136, 45), width=3)


class KiteCovariance(KiteView):
    title = 'Scene.covariance'

    def __init__(self, spool):
        model = spool.model

        covariance_plot = KiteNoisePlot(model)
        self.main_widget = covariance_plot
        self.tools = {
            'Covariance.powerspecNoise':
                KiteNoisePowerspec(covariance_plot),
            'Covariance.covariance_func':
                KiteCovariogram(covariance_plot),
            # 'Covariance.structure_func':
            #     KiteStructureFunction(covariance_plot),
        }

        self.param_covariance = KiteParamCovariance(model)
        self.parameters = [self.param_covariance]

        self.dialogInspectNoise = KiteToolNoise(model, spool)
        self.dialogInspectCovariance = KiteToolWeightMatrix(
            model, spool)

        spool.actionInspect_Noise.triggered.connect(
            self.dialogInspectNoise.show)
        covariance_plot.roi.sigClicked.connect(
            self.dialogInspectNoise.show)
        spool.actionInspect_Weights.triggered.connect(
            self.dialogInspectCovariance.show)
        spool.actionCalculate_WeightMatrix.triggered.connect(
            lambda: QCalculateWeightMatrix(model, spool))

        spool.actionInspect_Noise.setEnabled(True)
        spool.actionInspect_Weights.setEnabled(True)
        spool.actionCalculate_WeightMatrix.setEnabled(True)

        model.sigSceneModelChanged.connect(self.modelChanged)

        KiteView.__init__(self)

        for dock in self.tool_docks:
            dock.setStretch(10, .5)

    def modelChanged(self):
        self.dialogCovariance.close()
        self.main_widget.onConfigChanged()

        self.param_covariance.updateValues()
        for v in self.tools.itervalues():
            v.update()


class KiteNoisePlot(KitePlot):
    def __init__(self, model):
        self.components_available = {
            'displacement':
            ['Displacement', lambda sp: sp.scene.displacement],
        }
        self._component = 'displacement'

        KitePlot.__init__(self, model=model, los_arrow=True)

        llE, llN, sizeE, sizeN = self.model.covariance.noise_coord
        self.roi = pg.RectROI(
            pos=(llE, llN),
            size=(sizeE, sizeN),
            sideScalers=True,
            pen=pen_roi)

        self.roi.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.roi.sigRegionChangeFinished.connect(self.updateNoiseRegion)
        self.addItem(self.roi)

        self.model.sigCovarianceConfigChanged.connect(
            self.onConfigChanged)

    def onConfigChanged(self):
        llE, llN, sizeE, sizeN = self.model.covariance.noise_coord
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
        self.model.covariance.noise_coord = (llE, llN, sizeE, sizeN)
        self.model.covariance.noise_data = data.T


class _KiteSubplotPlot(QtGui.QWidget):
    def __init__(self, parent_plot):
        QtGui.QWidget.__init__(self)
        self.parent_plot = parent_plot
        self.model = parent_plot.model

        self.plot = pg.PlotWidget(background='default')
        self.plot.showGrid(True, True, alpha=.5)
        self.plot.setMenuEnabled(False)
        self.plot.enableAutoRange()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plot)

    def addItem(self, *args, **kwargs):
        self.plot.addItem(*args, **kwargs)


class KiteNoisePowerspec(_KiteSubplotPlot):
    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)

        self.power = pg.PlotDataItem(antialias=True)
        self.power_lin = pg.PlotDataItem(antialias=True, pen=pen_green_dash)

        self.power.setZValue(10)
        self.plot.setLabels(
            bottom='Wavenumber (cycles/m)',
            left='Power (m<sup>2</sup>)')

        self.plot.setLogMode(x=True, y=True)

        self.legend = pg.LegendItem(offset=(0., .5))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.power_lin, 'Log-linear model')

        self.addItem(self.power)
        # self.addItem(self.power_lin)

        self.model.sigCovarianceChanged.connect(self.update)
        self.update()

    def update(self):
        covariance = self.model.covariance
        spec, k, _, _, _, _ = covariance.powerspecNoise1D()
        self.power.setData(k, spec)
        self.power_lin.setData(
            k, covariance.powerspecModel(k))


class KiteCovariogram(_KiteSubplotPlot):
    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom={'Distance', 'm'},
                            left='Covariance (m<sup>2</sup>)')

        self.cov = pg.PlotDataItem(antialias=True)
        self.cov.setZValue(10)
        self.cov_model = pg.PlotDataItem(antialias=True, pen=pen_red_dot)
        self.variance = pg.InfiniteLine(
            pen=pen_variance,
            angle=0, movable=True, hoverPen=None,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.)),
                       'color': pg.mkColor(255, 255, 255, 155)})

        self.variance.sigPositionChangeFinished.connect(self.setVariance)

        self.addItem(self.variance)

        self.addItem(self.cov)
        self.addItem(self.cov_model)
        # self.cov_lin_pow = pg.PlotDataItem(antialias=True,
        #                                    pen=pen_green_dash)
        # self.addItem(self.cov_lin_pow)

        self.legend = pg.LegendItem(offset=(0., .5))

        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_model, '')
        self.legend.template = 'Model: {0:.5f} e^(-d/{1:.1f}) | RMS: {rms:.4e}'

        self.model.sigCovarianceChanged.connect(
            self.update)

        self.update()

    def setVariance(self):
        self.model.covariance.variance = self.variance.value()

    def update(self):
        covariance = self.model.covariance
        cov, dist = covariance.covariance_func

        self.cov.setData(dist, cov)
        self.cov_model.setData(
            dist, modelCovariance(dist, *covariance.covariance_model))
        # self.cov_lin_pow.setData(
        #     dist, covariance.covarianceAnalytical(3)[0])

        self.legend.items[-1][1].setText(
            self.legend.template.format(
                *covariance.covariance_model,
                rms=covariance.covariance_model_rms))
        self.variance.setValue(covariance.variance)


class KiteStructureFunction(_KiteSubplotPlot):
    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)

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
        self.model.sigCovarianceChanged.connect(
            self.update)
        self.variance.sigPositionChangeFinished.connect(
            self.changeVariance)

        self.update()

    def update(self):
        covariance = self.model.covariance
        struc, dist = covariance.structure_func
        self.structure.setData(dist, struc)
        self.variance.setValue(covariance.variance)

    def changeVariance(self, inf_line):
        covariance = self.model.covariance
        covariance.variance = inf_line.getYPos()


class KiteToolNoise(QtGui.QDialog):
    class NoisePlot(KitePlot):
        def __init__(self, model):
            self.components_available = {
                'noise_data': [
                  'Displacement',
                  lambda sp: self.noise_data_masked(sp.covariance)
                ]}

            self._component = 'noise_data'
            KitePlot.__init__(self, model=model)

        @staticmethod
        def noise_data_masked(covariance):
            data = covariance.noise_data.copy()
            data[data == 0.] = num.nan
            return data

        def proxy_connect(self):
            self.model.sigCovarianceChanged.connect(self.update)

        def proxy_disconnect(self):
            self.model.sigCovarianceChanged.disconnect(self.update)

    class NoiseSyntheticPlot(KitePlot):
        def __init__(self, model):
            sp = model
            _, _, sizeE, sizeN = sp.covariance.noise_coord

            self.patch_size_roi = pg.RectROI(
                pos=(0., 0.),
                size=(sizeE, sizeN),
                sideScalers=True,
                movable=False,
                pen=pen_roi)
            self.patch_size_roi.sigRegionChangeFinished.connect(self.update)

            self._anisotropic = False
            self.components_available = {
                'synthetic_noise': [
                  'Noise',
                  lambda sp: sp.covariance.syntheticNoise(
                    self.sizePatchPx(), anisotropic=self.anisotropic)
                ]}

            self._component = 'synthetic_noise'
            KitePlot.__init__(self, model=model)

            self.addItem(self.patch_size_roi)

        @property
        def anisotropic(self):
            return self._anisotropic

        def enableAnisotropic(self, value):
            self._anisotropic = value
            self.update()

        def resetSize(self):
            _, _, sizeE, sizeN = self.model.covariance.noise_coord
            self.patch_size_roi.setSize((sizeE, sizeN))

        def sizePatchPx(self):
            sp = self.model
            sizeE, sizeN = self.patch_size_roi.size()
            return int(sizeN / sp.frame.dN), int(sizeE / sp.frame.dE)

        def setGradientEditor(self, gradient_editor):
            ge = gradient_editor
            image = self.image

            hist_pen = pg.mkPen((170, 57, 57, 255), width=1.)
            image.setLookupTable(ge.getLookupTable)

            def updateLevels():
                image.setLevels(ge.region.getRegion())

            ge.sigLevelChangeFinished.connect(updateLevels)
            ge.sigLevelsChanged.connect(updateLevels)
            updateLevels()

            def updateHistogram():
                h = image.getHistogram()
                if h[0] is None:
                    return
                ge.hist_syn.setData(*h)

            ge.hist_syn = pg.PlotDataItem(pen=hist_pen)
            ge.hist_syn.rotate(90.)
            ge.vb.addItem(ge.hist_syn)
            updateHistogram()

            image.sigImageChanged.connect(updateHistogram)

        def proxy_connect(self):
            self.model.sigCovarianceChanged.connect(self.update)

        def proxy_disconnect(self):
            self.model.sigCovarianceChanged.disconnect(self.update)

    def __init__(self, model, parent=None):
        QtGui.QDialog.__init__(self, parent)

        loadUi(get_resource('noise_dialog.ui'), baseinstance=self)
        self.closeButton.setIcon(
            self.style().standardPixmap(QtGui.QStyle.SP_DialogCloseButton))
        self.setWindowFlags(QtCore.Qt.Window)

        self.noise_patch = self.NoisePlot(model)
        self.noise_synthetic = self.NoiseSyntheticPlot(model)

        colormap = KiteToolColormap(self.noise_patch)
        self.noise_synthetic.setGradientEditor(colormap)

        self.dockarea = dockarea.DockArea(self)

        self.dockarea.addDock(
            dockarea.Dock(
                'Covariance.noise_data',
                widget=self.noise_patch,
                size=(6, 6),
                autoOrientation=False,),
            position='top')

        self.dockarea.addDock(
            dockarea.Dock(
                'Covariance.syntheticNoise',
                widget=self.noise_synthetic,
                size=(6, 6),
                autoOrientation=False,),
            position='bottom')

        self.dockarea.addDock(
            dockarea.Dock(
                'Colormap',
                widget=colormap,
                size=(1, 1),
                autoOrientation=False,),
            position='right')
        self.horizontalLayoutPlot.addWidget(self.dockarea)

        self.resetSizeButton.released.connect(self.noise_synthetic.resetSize)
        self.anisotropicCB.toggled.connect(
            lambda b: self.noise_synthetic.enableAnisotropic(b))

    def closeEvent(self, ev):
        self.noise_patch.proxy_disconnect()
        self.noise_synthetic.proxy_disconnect()
        ev.accept()

    def showEvent(self, ev):
        self.noise_patch.update()
        self.noise_patch.proxy_connect()
        self.noise_synthetic.update()
        self.noise_synthetic.proxy_connect()
        ev.accept()


class KiteToolWeightMatrix(QtGui.QDialog):
    class MatrixPlot(KitePlot):
        def __init__(self, model):
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            self._component = 'weight'

            KitePlot.__init__(self, model)
            self.model = model

            gradient = Gradients['thermal']

            self.cmap = pg.ColorMap(
                pos=[c[0] for c in gradient['ticks']],
                color=[c[1] for c in gradient['ticks']],
                mode=gradient['mode'])
            self.image.setLookupTable(self.cmap.getLookupTable())

            self.setLabels(
                bottom={'Leaf #', ''},
                left={'Leaf #', ''})

            self.setAspectLocked(True)
            self.setMouseEnabled(x=False, y=False)

            self.hint = {
                'leaf1': 0,
                'leaf2': 0,
                'weight': num.nan}

            self.hint_text.template =\
                '<span style="font-family: monospace; color: #fff;'\
                'background-color: #000;">'\
                'Leaf #1: {leaf1:d} | Leaf #2: {leaf2:d} | '\
                'Weight {weight:e}</span>'

            self.update()

        def update(self):
            self.image.updateImage(
                self.model.covariance.weight_matrix_focal.T,
                autoLevels=True)

        def transFromFrame(self):
            # self.resetTransform()
            self.setRange(
                xRange=(0, self.model.quadtree.nleaves),
                yRange=(0, self.model.quadtree.nleaves))

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
            self.model.sigCovarianceChanged.connect(self.update)
            self.model.sigQuadtreeChanged.connect(self.transFromFrame)

        def proxy_disconnect(self):
            self.model.sigCovarianceChanged.disconnect(self.update)
            self.model.sigQuadtreeChanged.disconnect(self.transFromFrame)

    def __init__(self, model, parent=None):
        QtGui.QDialog.__init__(self, parent)

        loadUi(get_resource('covariance_matrix.ui'), baseinstance=self)
        self.closeButton.setIcon(
            self.style().standardPixmap(QtGui.QStyle.SP_DialogCloseButton))

        self.weight_matrix = self.MatrixPlot(model)
        self.dockarea = dockarea.DockArea(self)

        self.dockarea.addDock(
            dockarea.Dock(
                'Covariance.weight_matrix_focal',
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


class QCalculateWeightMatrix(QtCore.QObject):
    sigCalculateWeightMatrix = QtCore.Signal()

    def __init__(self, model, parent):
        QtCore.QObject.__init__(self)
        self.sigCalculateWeightMatrix.connect(
            model.calculateWeightMatrix)

        ret = QtGui.QMessageBox.information(
            parent,
            'Calculate full weight matrix',
            '''<html><head/><body><p>
This will calculate the quadtree's full weight matrix
(<span style='font-family: monospace'>Covariance.weight_matrix</span>)
for this noise/covariance configuration.</p><p>
The calculation is expensive and may take several minutes.
</p></body></html>
''', buttons=(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel))
        if ret == QtGui.QMessageBox.Ok:
            self.sigCalculateWeightMatrix.emit()


class KiteParamCovariance(KiteParameterGroup):
    def __init__(self, model, **kwargs):
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

        model.sigCovarianceChanged.connect(self.updateValues)
        KiteParameterGroup.__init__(
            self,
            model=model,
            model_attr='covariance',
            **kwargs)
