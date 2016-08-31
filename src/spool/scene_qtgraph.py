#!/bin/python
from PySide import QtGui
from PySide import QtCore
from qt_utils import QDoubleSlider

import numpy as num

import pyqtgraph as pg
from pyqtgraph import dockarea


class _QKitePlot(pg.PlotWidget):
    def __init__(self):
        pg.PlotWidget.__init__(self)

        # self.components_available = {
        # }

        # self._component = None

        self.image = pg.ImageItem(None)
        self.image.setCompositionMode(
            QtGui.QPainter.CompositionMode_SourceOver)
        self.setAspectLocked(True)

        self.addItem(self.image)
        self.update()

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        if component not in self.components_available.keys():
            raise AttributeError('Invalid component %s' % component)
        self._component = component
        self.update()

    @property
    def data(self):
        return self.components_available[self.component][1](self.container)

    def update(self):
        lmax, lmin = num.nanmax(self.data), num.nanmin(self.data)
        lvl = max(abs(lmax), abs(lmin))

        self.image.setImage(self.data, levels=(-lvl, lvl), autoDownsample=True)
        # self.addIsocurves()

    def addIsocurves(self):
        self.isocurves = pg.IsocurveItem(None)
        self.isocurves.setData(self.data)
        self.isocurves.setLevel(0.01)
        self.addItem(self.isocurves)


class QKiteDisplacementPlot(_QKitePlot):
    def __init__(self, scene):

        self.components_available = {
            'displacement': ['LOS Displacement', lambda sc: sc.displacement],
            'theta': ['LOS Theta', lambda sc: sc.theta],
            'phi': ['LOS Phi', lambda sc: sc.phi],
            'dE': ['Displacement dE', lambda sc: sc.cartesian.dE],
            'dN': ['Displacement dN', lambda sc: sc.cartesian.dN],
            'dU': ['Displacement dU', lambda sc: sc.cartesian.dU],
        }
        self._component = 'displacement'
        self.container = scene

        _QKitePlot.__init__(self)


class QKiteQuadtreePlot(_QKitePlot):
    def __init__(self, quadtree):

        self.components_available = {
            'mean': ['Mean Displacement', lambda qt: qt.leaf_matrix_means],
            'median': ['Median Theta', lambda qt: qt.leaf_matrix_medians],
        }
        self._component = 'mean'
        self.container = quadtree

        self.container.subscribe(self.update)

        _QKitePlot.__init__(self)


class QKiteToolComponents(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.plot = plot

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

            btn.setText(self.plot.components_available[comp][0])
            btn.setToolTip('Scene.%s' % comp)
            btn.setCheckable(True)
            btn.setChecked(comp == self.plot.component)

            btn.clicked_str = QtCore.Signal(str)
            btn.clicked.connect(partial(changeComponent, comp))

        layout.addStretch(3)
        return layout


class QKiteToolGradient(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        self.gradient = pg.GradientWidget(orientation='top')
        self.gradient.restoreState({'ticks':
                                    [(0., (106, 0, 31, 255)),
                                     (.5, (255, 255, 255, 255)),
                                     (1., (8, 54, 104, 255))],
                                    'mode': 'rgb'})

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getGradientWidget())
        self.layout.addWidget(self.getResetButton())
        self.layout.addStretch(3)

    def getGradientWidget(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.gradient)

        def updatePlot():
            self.plot.image.setLookupTable(
                self.gradient.getLookupTable(100, alpha=True))
        self.gradient.sigGradientChanged.connect(updatePlot)

        group = QtGui.QGroupBox('Control Colormap')
        group.setLayout(layout)

        return group

    def getResetButton(self):
        btn = QtGui.QPushButton('Reset Colormap')

        def resetColormap():
            self.gradient.restoreState({'ticks':
                                        [(0., (106, 0, 31, 255)),
                                         (.5, (255, 255, 255, 255)),
                                         (1., (8, 54, 104, 255))],
                                        'mode': 'rgb'})

        btn.released.connect(resetColormap)
        return btn


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.quadtree = plot.container

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getEpsilonChanger())
        self.layout.addWidget(self.getMethodsChanger())
        self.layout.addStretch(3)

    def getEpsilonChanger(self):
        layout = QtGui.QHBoxLayout()

        slider = QDoubleSlider(QtCore.Qt.Horizontal)
        spin = QtGui.QDoubleSpinBox()

        def changeEpsilon():
            self.quadtree.epsilon = spin.value()
            slider.setValue(spin.value())

        def updateRange():
            for wdgt in [slider, spin]:
                wdgt.setValue(self.quadtree.epsilon)
                wdgt.setRange(self.quadtree._epsilon_limit,
                              3*self.quadtree._epsilon)
                wdgt.setSingleStep(self.quadtree._epsilon/1e2)

        updateRange()
        spin.setDecimals(3)

        self.quadtree.splitMethodChanged.subscribe(updateRange)
        spin.valueChanged.connect(changeEpsilon)
        slider.valueChanged.connect(lambda: spin.setValue(slider.value()))

        layout.addWidget(spin)
        layout.addWidget(slider)

        group = QtGui.QGroupBox('Epsilon Control')
        group.setLayout(layout)

        return group

    def getMethodsChanger(self):
        from functools import partial

        layout = QtGui.QVBoxLayout()

        def changeMethod(method):
            self.quadtree.setSplitMethod(method)

        for method in self.quadtree._split_methods.keys():
            btn = QtGui.QRadioButton()
            btn.setText(self.quadtree._split_methods[method][0])
            btn.setChecked(method == self.quadtree.split_method)
            btn.clicked.connect(partial(changeMethod, method))

            layout.addWidget(btn)

        group = QtGui.QGroupBox('Split Method')
        group.setLayout(layout)

        return group


class QKiteDock(dockarea.DockArea):
    def __init__(self, container):
        dockarea.DockArea.__init__(self)

        main_dock = dockarea.Dock(self.title)
        main_widget = self.main_widget(container)
        main_dock.addWidget(main_widget)

        for i, (name, tool) in enumerate(self.tools.iteritems()):
            tool_dock = dockarea.Dock(name,
                                      widget=tool(main_widget),
                                      size=(5, 10))
            self.addDock(tool_dock, position='bottom')

        self.addDock(main_dock, position='left')


class QKiteDisplacementDock(QKiteDock):
    def __init__(self, scene):
        self.title = 'Displacement'
        self.main_widget = QKiteDisplacementPlot
        self.tools = {
            'Components': QKiteToolComponents,
            'Colormap': QKiteToolGradient,
        }

        QKiteDock.__init__(self, scene)


class QKiteQuadtreeDock(QKiteDock):
    def __init__(self, quadtree):
        self.title = 'Quadtree'
        self.main_widget = QKiteQuadtreePlot
        self.tools = {
            'Quadtree Control': QKiteToolQuadtree,
            'Components': QKiteToolComponents,
            'Colormap': QKiteToolGradient,
        }

        QKiteDock.__init__(self, quadtree)


if __name__ == '__main__':
    from kite.scene import SceneSynTest
    app = QtGui.QApplication(['Testing PyQtGraph'])
    sc = SceneSynTest.createGauss()

    win = QtGui.QMainWindow()
    win.resize(800, 800)

    img = QKiteDisplacementDock(sc)

    win.setCentralWidget(img)
    win.show()
    app.exec_()
