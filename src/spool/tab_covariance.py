import numpy as num
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes

from collections import OrderedDict

from PyQt5 import QtGui, QtCore
from pyqtgraph import dockarea

from kite.qt_utils import loadUi
from kite.covariance import CovarianceConfig

from .base import (KiteView, KitePlot, KiteParameterGroup,
                   KiteToolColormap, get_resource)

pen_covariance_model = pg.mkPen(
    (204, 0, 0), width=2, style=QtCore.Qt.DotLine)

pen_covariance = pg.mkPen(
    (255, 255, 255, 100), width=1.25)

pen_covariance_active = pg.mkPen(
    (255, 255, 255), width=1.25)

pen_variance = pg.mkPen(
    (78, 154, 6), width=2.5, style=QtCore.Qt.DashLine)
pen_variance_highlight = pg.mkPen(
    (115, 210, 22), width=2.5, style=QtCore.Qt.DashLine)

pen_green_dash = pg.mkPen(
    (115, 210, 22), width=2.5, style=QtCore.Qt.DashLine)

pen_roi = pg.mkPen(
    (78, 154, 6), width=2)
pen_roi_highlight = pg.mkPen(
    (115, 210, 22), width=2, style=QtCore.Qt.DashLine)


class KiteCovariance(KiteView):
    title = 'Scene.covariance'

    def __init__(self, spool):
        model = spool.model

        covariance_plot = KiteNoisePlot(model)
        self.main_widget = covariance_plot
        self.tools = {
            # 'Covariance.powerspecNoise':
            #     KiteNoisePowerspec(covariance_plot),
            'Semi-Variogram: Covariance.structure_spatial':
                KiteStructureFunction(covariance_plot),
            'Covariogram: Covariance.covariance_spatial':
                KiteCovariogram(covariance_plot),
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
        for v in self.tools.values():
            v.update()


class KiteNoisePlot(KitePlot):
    class NoisePatchROI(pg.RectROI):
        def _makePen(self):
            # Generate the pen color for this ROI based on its current state.
            if self.mouseHovering:
                return pen_roi_highlight
            else:
                return self.pen

    def __init__(self, model):
        self.components_available = {
            'displacement':
            ['Displacement', lambda sp: sp.scene.displacement],
        }
        self._component = 'displacement'

        KitePlot.__init__(self, model=model, los_arrow=True)

        llE, llN, sizeE, sizeN = self.model.covariance.noise_coord
        self.roi = self.NoisePatchROI(
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
        self.plot.setMenuEnabled(True)
        self.plot.enableAutoRange()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plot)

    def addItem(self, *args, **kwargs):
        self.plot.addItem(*args, **kwargs)


class KiteNoisePowerspec(_KiteSubplotPlot):
    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)

        self.power = pg.PlotDataItem(antialias=True)
        # self.power_lin = pg.PlotDataItem(antialias=True, pen=pen_green_dash)

        self.power.setZValue(10)
        self.plot.setLabels(
            bottom='Wavenumber (cycles/m)',
            left='Power (m<sup>2</sup>)')

        self.plot.setLogMode(x=True, y=True)

        # self.legend = pg.LegendItem(offset=(0., .5))
        # self.legend.setParentItem(self.plot.graphicsItem())
        # self.legend.addItem(self.power_lin, 'Log-linear model')

        self.addItem(self.power)
        # self.addItem(self.power_lin)

        self.model.sigCovarianceChanged.connect(self.update)
        self.update()

    @QtCore.pyqtSlot()
    def update(self):
        covariance = self.model.covariance
        spec, k, _, _, _, _ = covariance.powerspecNoise1D()
        self.power.setData(k, spec)
        # self.power_lin.setData(
        #     k, covariance.powerspecModel(k))


class KiteCovariogram(_KiteSubplotPlot):

    legend_template = {
        'exponential':
            'Model: {0:.2g} e^(-d/{1:.1f}) | RMS: {rms:.4e}',
        'exponential_cosine':
            'Model: {0:.2g} e^(-d/{1:.1f}) - cos((d-({2:.1f}))/{3:.1f}) '
            '| RMS: {rms:.4e}'
    }

    class VarianceLine(pg.InfiniteLine):
        def __init__(self, *args, **kwargs):
            pg.InfiniteLine.__init__(self, *args, **kwargs)
            self.setCursor(QtCore.Qt.SizeVerCursor)

    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)
        self.plot.setLabels(bottom=('Distance', 'm'),
                            left='Covariance (m<sup>2</sup>)')

        self.cov_spectral = pg.PlotDataItem(antialias=True)
        self.cov_spectral.setZValue(10)

        self.cov_spatial = pg.PlotDataItem(antialias=True)

        self.cov_model = pg.PlotDataItem(
            antialias=True,
            pen=pen_covariance_model)

        self.variance = self.VarianceLine(
            pen=pen_variance,
            angle=0, movable=True, hoverPen=pen_variance_highlight,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.)),
                       'color': pg.mkColor(255, 255, 255, 155)})
        self.variance.setToolTip('Move to change variance')
        self.variance.sigPositionChangeFinished.connect(self.setVariance)

        self.addItem(self.cov_spectral)
        self.addItem(self.cov_spatial)
        self.addItem(self.cov_model)
        self.addItem(self.variance)
        # self.cov_lin_pow = pg.PlotDataItem(antialias=True,
        #                                    pen=pen_green_dash)
        # self.addItem(self.cov_lin_pow)

        self.legend = pg.LegendItem(offset=(0., .5))

        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.addItem(self.cov_model, '')

        self.model.sigCovarianceChanged.connect(
            self.update)

        self.update()

    def setVariance(self):
        self.model.covariance.variance = self.variance.value()

    @QtCore.pyqtSlot()
    def update(self):
        covariance = self.model.covariance

        cov_spectral, dist = covariance.covariance_spectral
        self.cov_spectral.setData(dist, cov_spectral)

        cov_spatial, dist = covariance.covariance_spatial
        self.cov_spatial.setData(dist, cov_spatial)

        if self.model.covariance.config.sampling_method == 'spatial':
            self.cov_spatial.setPen(pen_covariance_active)
            self.cov_spectral.setPen(pen_covariance)

        else:
            self.cov_spatial.setPen(pen_covariance)
            self.cov_spectral.setPen(pen_covariance_active)

        model = self.model.covariance.getModelFunction()

        self.cov_model.setData(
            dist,
            model(dist, *covariance.covariance_model))

        tmpl = self.legend_template[covariance.config.model_function]

        self.legend.items[-1][1].setText(
            tmpl.format(
                *covariance.covariance_model,
                rms=covariance.covariance_model_rms))
        self.variance.setValue(covariance.variance)


class KiteStructureFunction(_KiteSubplotPlot):

    class VarianceLine(pg.InfiniteLine):
        def __init__(self, *args, **kwargs):
            pg.InfiniteLine.__init__(self, *args, **kwargs)
            self.setCursor(QtCore.Qt.SizeVerCursor)

    def __init__(self, parent_plot):
        _KiteSubplotPlot.__init__(self, parent_plot)

        self.structure = pg.PlotDataItem(
            antialias=True,
            pen=pen_covariance_active)
        self.variance = self.VarianceLine(
            pen=pen_variance,
            angle=0, movable=True, hoverPen=pen_variance_highlight,
            label='Variance: {value:.5f}',
            labelOpts={'position': .975,
                       'anchors': ((1., 0.), (1., 1.)),
                       'color': pg.mkColor(255, 255, 255, 155)})
        self.plot.setLabels(bottom=('Distance', 'm'),
                            left='Covariance (m<sup>2</sup>)')

        self.addItem(self.structure)
        self.addItem(self.variance)
        self.model.sigCovarianceChanged.connect(
            self.update)
        self.variance.sigPositionChangeFinished.connect(
            self.changeVariance)

        self.update()

    @QtCore.pyqtSlot()
    def update(self):
        covariance = self.model.covariance
        struc, dist = covariance.getStructure()
        self.structure.setData(dist[num.isfinite(struc)],
                               struc[num.isfinite(struc)])
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

        @QtCore.pyqtSlot()
        def update(self):
            KitePlot.update(self)

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

            self._anisotropic = False
            self.components_available = {
                'synthetic_noise': [
                  'Noise',
                  lambda sp: sp.covariance.syntheticNoise(
                    self.sizePatchPx(), anisotropic=self.anisotropic)
                ]}

            self._component = 'synthetic_noise'

            KitePlot.__init__(self, model=model)
            self.patch_size_roi.sigRegionChangeFinished.connect(self.update)
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

        @QtCore.pyqtSlot()
        def update(self):
            KitePlot.update(self)

        def proxy_disconnect(self):
            self.model.sigCovarianceChanged.disconnect(self.update)

    def __init__(self, model, parent=None):
        QtGui.QDialog.__init__(self, parent)

        loadUi(get_resource('noise_dialog.ui'), baseinstance=self)
        self.closeButton.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
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

        @QtCore.pyqtSlot()
        def update(self):
            self.image.updateImage(
                self.model.covariance.weight_matrix_focal.T,
                autoLevels=True)

        def transFromFrame(self):
            # self.resetTransform()
            self.setRange(
                xRange=(0, self.model.quadtree.nleaves),
                yRange=(0, self.model.quadtree.nleaves))

        @QtCore.pyqtSlot(object)
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
            self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))

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
    sigCalculateWeightMatrix = QtCore.pyqtSignal()

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
    sigSamplingMethod = QtCore.pyqtSignal(str)
    sigSpatialBins = QtCore.pyqtSignal(int)
    sigSpatialPairs = QtCore.pyqtSignal(int)

    def __init__(self, model, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.covariance'

        self.sp = model

        self.parameters = OrderedDict([
            ('variance', None),
            ('covariance_model',
             lambda c: ', '.join('%g' % p for p in c.covariance_model)),
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

        def changeSamplingMethod():
            model.covariance.setSamplingMethod(sampling_method.value())

        p = {'name': 'sampling_method',
             'values': {
                'spatial random': 'spatial',
                'spectral': 'spectral',
                 },
             'value': model.covariance.config.sampling_method,
             'tip': CovarianceConfig.sampling_method.help,
             }
        sampling_method = pTypes.ListParameter(**p)
        sampling_method.sigValueChanged.connect(changeSamplingMethod)

        def changeSpatialBins():
            model.covariance.setSpatialBins(spatial_bins.value())

        p = {'name': 'spatial_bins',
             'value': model.covariance.config.spatial_bins,
             'type': 'int',
             'limits': (25, 500),
             'step': 5,
             'edditable': True,
             'tip': CovarianceConfig.spatial_bins.help
             }

        spatial_bins = pTypes.SimpleParameter(**p)
        spatial_bins.sigValueChanged.connect(changeSpatialBins)

        def changeSpatialPairs():
            model.covariance.setSpatialPairs(spatial_pairs.value())

        p = {'name': 'spatial_pairs',
             'value': model.covariance.config.spatial_pairs,
             'type': 'int',
             'limits': (0, 1000000),
             'step': 50000,
             'edditable': True,
             'tip': CovarianceConfig.spatial_pairs.help
             }

        spatial_pairs = pTypes.SimpleParameter(**p)
        spatial_pairs.sigValueChanged.connect(changeSpatialPairs)

        def changeModelFunction():
            model.covariance.setModelFunction(model_function.value())

        p = {'name': 'model_function',
             'values': {
                'exponential': 'exponential',
                'exp + cosine': 'exponential_cosine',
                 },
             'value': model.covariance.config.model_function,
             'tip': CovarianceConfig.model_function.help
             }
        model_function = pTypes.ListParameter(**p)
        model_function.sigValueChanged.connect(changeModelFunction)

        self.pushChild(model_function)
        self.pushChild(spatial_bins)
        self.pushChild(spatial_pairs)
        self.pushChild(sampling_method)
