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


class QKiteDisplacementPlot2D(FigureCanvas):
    def __init__(self, parent=None, plot=None):
        self.fig = Figure(frameon=False, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # self.axes.hold(False)
        self.axes = self.fig.add_subplot(111)

        self.plot = plot
        self.plot.plot(axes=self.axes)

        self.axes.set_title('')

        def updatePlot():
            self.fig.canvas.draw()

        self.plot.subscribe(updatePlot)


class QKiteQuadtreePlot2D(QKiteDisplacementPlot2D):
    def __init__(self, parent=None, plot=None):
        QKiteDisplacementPlot2D.__init__(self, parent, plot)

        def updatePlot():
            self.plot._update()

        self.quadtree = plot._quadtree
        self.quadtree.subscribe(updatePlot)


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, parent=None, plot=None):
        QtGui.QWidget.__init__(self, parent)
        self.quadtree = plot._quadtree

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(QtGui.QLabel('Epsilon'))
        self.layout.addLayout(self.getEpsilonValueLayout())
        self.layout.addLayout(self.getMethodLayout())
        self.layout.addStretch(3)

    def getEpsilonValueLayout(self):
        layout = QtGui.QHBoxLayout()

        slider = QDoubleSlider(QtCore.Qt.Horizontal)
        spin = QtGui.QDoubleSpinBox()

        for wdgt in [slider, spin]:
            wdgt.setValue(self.quadtree.epsilon)
            wdgt.setRange(self.quadtree._epsilon - self.quadtree._epsilon,
                          self.quadtree._epsilon + self.quadtree._epsilon)
            wdgt.setSingleStep(self.quadtree._epsilon/1e2)
        spin.setDecimals(3)

        def changeEpsilon():
            self.quadtree.epsilon = spin.value()
            slider.setValue(spin.value())

        spin.valueChanged.connect(changeEpsilon)
        slider.valueChanged.connect(lambda: spin.setValue(slider.value()))

        layout.addWidget(spin)
        layout.addWidget(slider)

        return layout

    def getMethodLayout(self):
        from functools import partial

        layout = QtGui.QVBoxLayout()

        def changeMethod(method):
            self.quadtree.setSplitMethod(method)

        for method in self.quadtree._split_methods.keys():
            btn = QtGui.QRadioButton()
            btn.setText(method)
            btn.setChecked(method == self.quadtree.split_method)
            btn.clicked.connect(partial(changeMethod, method))

            layout.addWidget(btn)

        return layout


class QKiteToolColormap(QtGui.QWidget):
    def __init__(self, parent=None, plot=None):
        QtGui.QWidget.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        layout = QtGui.QVBoxLayout(self)

        layout.addLayout(self.getColormapsLayout())
        layout.addWidget(self.getColormapAdjustButton())
        layout.addWidget(self.getColormapSymmetricButton())
        layout.addLayout(self.getColormapRange())

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

        grid_layout = QtGui.QGridLayout()
        grid_layout.addWidget(QtGui.QLabel('Max'), 1, 1)
        grid_layout.addWidget(max_spin, 1, 2)
        grid_layout.addWidget(max_slider, 1, 3)

        grid_layout.addWidget(QtGui.QLabel('Min'), 2, 1)
        grid_layout.addWidget(min_spin, 2, 2)
        grid_layout.addWidget(min_slider, 2, 3)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(grid_layout)
        layout.addStretch(3)

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


class QKiteToolComponents(QtGui.QWidget):
    def __init__(self, parent=None, plot=None):
        QtGui.QWidget.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        self.setLayout(self.getComponentsLayout())

    def getComponentsLayout(self):
        from functools import partial

        layout = QtGui.QVBoxLayout()
        self._btn_grp = QtGui.QButtonGroup()
        self._btn_grp.setExclusive(True)

        def changeComponent(component):
            self.plot.component = component

        components = self.plot.components_available.keys()
        components.sort()
        for comp in components:
            btn = QtGui.QPushButton(self)

            layout.addWidget(btn)
            self._btn_grp.addButton(btn)

            btn.setText(self.plot.components_available[comp]['name'])
            btn.setToolTip('Scene.%s' % comp)
            btn.setCheckable(True)
            btn.setChecked(comp == self.plot.component)

            btn.clicked_str = QtCore.Signal(str)
            btn.clicked.connect(partial(changeComponent, comp))

        layout.addStretch(3)
        return layout


class QKiteMetaTable(QtGui.QTableWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QTableWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        self.populateTable()

    def populateTable(self):
        for row, (k, v) in enumerate(
                            self.scene.meta.T.inamevals(self.scene.meta)):
            key = QtGui.QTableWidgetItem(k)
            value = QtGui.QTableWidgetItem(v)

            self.setItem(row, 1, key)
            self.setItem(row, 2, value)

'''
Abstract tab classes
'''


class QKiteTab(QtGui.QSplitter, _QKiteScene):
    def __init__(self, parent=None, scene=None, plot=None):
        QtGui.QSplitter.__init__(self, parent)
        self.scene = scene
        self.plot = plot

        self.plot_properties = QtGui.QSplitter(QtCore.Qt.Vertical)

        self.toolbox = QtGui.QToolBox(self.plot_properties)
        self.plot_properties.addWidget(self.toolbox)
        self.plot_properties.addWidget(QKiteMetaTable(
                                       self.plot_properties, self.scene))

        self.initMain()
        self.populateToolbox()

        self.addWidget(self.main)
        self.addWidget(self.plot_properties)

    def populateToolbox(self):
        for name, tool in self.tools.iteritems():
            self.toolbox.addItem(tool(self, self.plot), name)

    def initMain(self):
        self.main = self.main_widget(self, self.plot)


class QKiteDisplacementTab(QKiteTab):
    def __init__(self, parent=None, scene=None):
        self.title = 'Displacement'
        self.main_widget = QKiteDisplacementPlot2D
        self.tools = {
            'Components': QKiteToolComponents,
            'Colormap': QKiteToolColormap,
        }

        QKiteTab.__init__(self, parent, scene, scene.plot)


class QKiteQuadtreeTab(QKiteTab):
    def __init__(self, parent, scene):
        self.title = 'Quadtree'
        self.main_widget = QKiteQuadtreePlot2D

        self.tools = {
                'Quadtree': QKiteToolQuadtree,
                'Components': QKiteToolComponents,
                'Colormap': QKiteToolColormap,
        }

        QKiteTab.__init__(self, parent, scene, scene.quadtree.plot)


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()

    qt_app = QtGui.QApplication(['KiteSpool'])
    app = QtGui.QMainWindow()

    tabs = QtGui.QTabWidget(app)
    s = QKiteDisplacementTab(tabs, scene=sc)
    tabs.addTab(s, 'Scene')
    tabs.setGeometry(1, 1, 400, 400)

    app.show()
    qt_app.exec_()
