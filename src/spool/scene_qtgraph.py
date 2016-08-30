#!/bin/python
from PySide import QtGui
from PySide import QtCore
# import numpy as num

import pyqtgraph as pg
from pyqtgraph import dockarea


class QKiteDisplacementPlot(pg.PlotWidget):
    def __init__(self, scene):
        pg.PlotWidget.__init__(self)

        self.components_available = {
            'displacement': {'name': 'LOS Displacement',
                             'eval': lambda sc: sc.displacement},
            'theta': {'name': 'LOS Theta',
                      'eval': lambda sc: sc.theta},
            'phi': {'name': 'LOS Phi',
                    'eval': lambda sc: sc.phi},
            'dE': {'name': 'Displacement dE',
                   'eval': lambda sc: sc.cartesian.dE},
            'dN': {'name': 'Displacement dN',
                   'eval': lambda sc: sc.cartesian.dN},
            'dU': {'name': 'Displacement dU',
                   'eval': lambda sc: sc.cartesian.dU},
        }
        self._scene = scene
        self._component = 'displacement'

        self.image = pg.ImageItem(self.data)
        self.setAspectLocked(True)

        self.addItem(self.image)

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
        return self.components_available[self.component]['eval'](self._scene)

    def update(self):
        print self.data
        self.image.setImage(self.data)


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

            btn.setText(self.plot.components_available[comp]['name'])
            btn.setToolTip('Scene.%s' % comp)
            btn.setCheckable(True)
            btn.setChecked(comp == self.plot.component)

            btn.clicked_str = QtCore.Signal(str)
            btn.clicked.connect(partial(changeComponent, comp))

        layout.addStretch(3)
        return layout


class QKiteToolGradient(pg.GradientWidget):
    """docstring for QKiteToolGradient"""
    def __init__(self, plot):
        pg.GradientWidget.__init__(self, orientation='top')

        def updatePlot(self):
            plot.image.setLookupTable(self.getLookupTable(200))
        self.sigGradientChanged.connect(updatePlot)


class QKiteDock(dockarea.DockArea):
    def __init__(self, scene):
        dockarea.DockArea.__init__(self)

        main_dock = dockarea.Dock(self.title)
        main_widget = self.main_widget(scene)
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
