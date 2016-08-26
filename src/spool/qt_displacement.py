#!/bin/python
from PySide import QtGui
from PySide import QtCore
from qt_utils import QDoubleSlider
import numpy as num
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg \
                                        as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['backend.qt4'] = 'PySide'


class _QKiteScene(object):
    def __init__(self, scene=None):
        if scene is None:
            try:
                if self.parent().__dict__.get('scene', None) is not None:
                    self.scene = self.parent().__dict__.get('scene')
                    return
            except Exception as e:
                print e
        else:
            self.scene = scene
            return
        raise AttributeError('Scene must be defined')


class QKiteScenePlot2D(FigureCanvas, _QKiteScene):
    def __init__(self, parent=None, scene=None):

        self.fig = Figure(frameon=False, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        _QKiteScene.__init__(self, scene)
        # self.axes.hold(False)
        self.axes = self.fig.add_subplot(111)

        self.plot = self.scene.plot
        self.plot.plot(axes=self.axes)

        def updatePlot():
            self.fig.canvas.draw()

        self.plot.subscribe(updatePlot)


class QKiteControlScene(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        # QKiteControlColormap(self, self.scene.plot)
        QKiteControlComponent(self, self.scene.plot)


class QKiteControlColormap(QtGui.QVBoxLayout):
    def __init__(self, parent=None, plot=None):
        QtGui.QVBoxLayout.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        self.addLayout(self.getColormapsLayout())
        self.addWidget(self.getColormapAdjustButton())
        self.addWidget(self.getColormapSymmetricButton())
        self.addLayout(self.getColormapRange())

    def getColormapsLayout(self):
        cm_colormap = QtGui.QHBoxLayout()
        cm_combo = QtGui.QComboBox()
        for cmtype, cmlist in self.plot._colormapsAvailable():
            cm_combo.addItem('%s' % cmtype)
            _i = cm_combo.findText('%s' % cmtype)
            cm_combo.setItemData(_i, '', role=QtCore.Qt.UserRole-1)

            for cm in cmlist:
                cm_combo.addItem(' %s' % cm, cm)

        _i = cm_combo.findData(self.plot.image.get_cmap().name)
        cm_combo.setCurrentIndex(_i)

        def changeColormap(index):
            self.plot.setColormap(cm_combo.itemData(index))

        cm_combo.currentIndexChanged.connect(changeColormap)

        cm_colormap.addWidget(QtGui.QLabel('Colormap'))
        cm_colormap.addWidget(cm_combo)
        return cm_colormap

    def getColormapRange(self):
        layout = QtGui.QGridLayout()

        def _getSpinBox():
            ''' Simple generator DRY '''
            sb = QtGui.QDoubleSpinBox()
            sb.setDecimals(3)
            sb.setSingleStep(.05)
            return sb

        def _getSlider():
            sl = QDoubleSlider(QtCore.Qt.Horizontal)
            sl.setSingleStep(5)
            return sl

        max_spin = _getSpinBox()
        max_slider = _getSlider()
        min_spin = _getSpinBox()
        min_slider = _getSlider()

        min_spin.setValue(self.plot.colormap_limits[0])
        min_slider.setValue(self.plot.colormap_limits[0])
        max_spin.setValue(self.plot.colormap_limits[1])
        max_slider.setValue(self.plot.colormap_limits[1])

        def updateRanges():
            rmax, rmin = num.nanmax(self.plot.data), num.nanmin(self.plot.data)
            for w in [max_spin, max_slider, min_slider, min_spin]:
                w.setRange(rmin-rmin*.5, rmax+rmax*.5)

        updateRanges()

        def fromSpinRange():
            vmax, vmin = max_spin.value(), min_spin.value()
            self.plot.colormap_limits = (vmin, vmax)
            return

        max_spin.valueChanged.connect(fromSpinRange)
        max_slider.valueChanged.connect(
            lambda: max_spin.setValue(max_slider.value()))

        min_spin.valueChanged.connect(fromSpinRange)
        min_slider.valueChanged.connect(
            lambda: min_spin.setValue(min_slider.value()))

        layout.addWidget(QtGui.QLabel('Max'), 1, 1)
        layout.addWidget(max_spin, 1, 2)
        layout.addWidget(max_slider, 1, 3)

        layout.addWidget(QtGui.QLabel('Min'), 2, 1)
        layout.addWidget(min_spin, 2, 2)
        layout.addWidget(min_slider, 2, 3)
        return layout

    def getColormapAdjustButton(self):
        cm_auto = QtGui.QPushButton()
        cm_auto.setText('Adjust range')

        def adjustColormap():
            self.plot.colormapAdjust()
        cm_auto.released.connect(adjustColormap)
        return cm_auto

    def getColormapSymmetricButton(self):
        cm_sym = QtGui.QPushButton()
        cm_sym.setText('Symetric colormap')
        cm_sym.setCheckable(True)
        cm_sym.setChecked(self.plot.colormap_symmetric)

        def updateSymmetry():
            self.plot.colormap_symmetric = cm_sym.isChecked()
        cm_sym.toggled.connect(updateSymmetry)
        return cm_sym


class QKiteControlComponent(QtGui.QVBoxLayout):
    def __init__(self, parent=None, plot=None):
        QtGui.QVBoxLayout.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        self.addLayout(self.getComponentsLayout())

    def getComponentsLayout(self):
        layout = QtGui.QVBoxLayout()
        self._btn_grp = QtGui.QButtonGroup()
        self._btn_grp.setExclusive(True)

        def changeComponent(component):
            print component
            self.plot.component = component

        components = self.plot.components_available.keys()
        components.sort()
        for comp in components:
            btn = QtGui.QPushButton()
            layout.addWidget(btn)
            self._btn_grp.addButton(btn)

            btn.setText(self.plot.components_available[comp])
            btn.setCheckable(True)
            btn.setChecked(comp == self.plot.component)
            btn.clicked.connect(lambda: changeComponent(comp))

        return layout


class QKiteScene(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        QKiteScenePlot2D(self)
        tab1 = QtGui.QDockWidget('Colorbar', self)
        tab1.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        tab1.setWidget(QKiteControlScene(parent=tab1, scene=scene))
        tab1.setFloating(True)
        tab1.setFeatures(QtGui.QDockWidget.DockWidgetFeature.DockWidgetMovable)

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()

    qt_app = QtGui.QApplication(['KiteSpool'])
    app = QKiteScene(scene=sc)
    app.show()
    qt_app.exec_()
