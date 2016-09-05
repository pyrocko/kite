#!/bin/python
from PySide import QtGui
from PySide import QtCore
from qt_utils import QDoubleSlider

import numpy as num

import pyqtgraph as pg
from pyqtgraph import dockarea


class _QKitePlot(pg.PlotWidget):
    def __init__(self, container):
        pg.PlotWidget.__init__(self)

        # self.components_available = {
        # }

        # self._component = None

        self.container = container

        self.image = pg.ImageItem(None)
        self.image.setCompositionMode(
            QtGui.QPainter.CompositionMode_SourceOver)

        self.setAspectLocked(True)
        self.setBackground((255, 255, 255, 255))

        self.addItem(self.image)
        self.update()

        # self.addIsocurve()
        # self.scalebar()

    def scalebar(self):
        ''' Not working '''
        self.scale_bar = pg.ScaleBar(10, width=5, suffix='m')
        self.scale_bar.setParentItem(self.plotItem)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-20, -20))

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
        self.image.setImage(self.data, autoDownsample=True)
        # self.addIsocurves()

    def addIsocurve(self, level=0.):
        iso = pg.IsocurveItem(level=level, pen='g')
        iso.setZValue(1000)
        iso.setData(pg.gaussianFilter(self.data, (5, 5)))
        iso.setParentItem(self.image)

        self.iso = iso


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

        _QKitePlot.__init__(self, container=scene)

        # ll_x, ll_y = scene.utm_x.min(), scene.utm_y.min()
        # ur_x, ur_y = scene.utm_x.max(), scene.utm_y.max()

        # scale_x = scene.utm_x.size/abs(ur_x - ll_x)
        # scale_y = scene.utm_y.size/abs(ur_y - ll_y)

        # self.image.translate(ll_x, ll_y)
        # self.image.scale(scale_x, scale_y)
        # self.setLimits(xMin=ll_x-(scale_x*scene.utm_x.size)*.2,
        #                xMax=ur_x+(scale_x*scene.utm_x.size)*.2,
        #                yMin=ll_y-(scale_y*scene.utm_y.size)*.2,
        #                yMax=ur_y+(scale_y*scene.utm_y.size)*.2)


class QKiteQuadtreePlot(_QKitePlot):
    def __init__(self, quadtree):

        self.components_available = {
            'mean': ['Mean Displacement', lambda qt: qt.leaf_matrix_means],
            'median': ['Median Theta', lambda qt: qt.leaf_matrix_medians],
        }
        self._component = 'mean'

        _QKitePlot.__init__(self, container=quadtree)
        self.container.subscribe(self.update)


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


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.quadtree = plot.container

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getEpsilonChanger())
        self.layout.addWidget(self.getMethodsChanger())
        self.layout.addWidget(self.getInfoPanel())
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
        group.setToolTip('''<p>Standard deviation/split
                        method of each tile is >= epsilon</p>''')
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

        group = QtGui.QGroupBox('Tile Split Criteria')
        group.setLayout(layout)

        return group

    def getInfoPanel(self):
        layout = QtGui.QVBoxLayout()
        info_text = QtGui.QLabel()

        def updateInfoText():
            info_text.setText('''
                <table><tr>
                <td style='padding-right: 10px'>Leaf count:</td>
                    <td><b>%d</b></td>
                </tr><tr>
                <td>Epsilon current:</td><td>%0.3f</td>
                </tr><tr>
                <td>Epsilon limit:</td><td>%0.3f</td>
                </tr></table>
                ''' % (len(self.quadtree.leafs), self.quadtree.epsilon,
                       self.quadtree._epsilon_limit))

        updateInfoText()
        self.quadtree.subscribe(updateInfoText)

        layout.addWidget(info_text)
        group = QtGui.QGroupBox('Quadtree Information')
        group.setLayout(layout)

        return group


class QKiteToolTransect(pg.PlotWidget):
    def __init__(self, plot):
        pg.PlotWidget.__init__(self)
        self.plot = plot

        self.transect = pg.PlotDataItem()
        self.addItem(self.transect)

        self.polyline = pg.PolyLineROI(positions=[(0, 0), (300, 300)],
                                       pen='g')
        self.plot.addItem(self.polyline)
        self.polyline.sigRegionChangeFinished.connect(self.updateTransect)

    def updateTransect(self):
        trans = num.ndarray((0))
        for line in self.polyline.segments:
            trans = num.append(trans,
                               line.getArrayRegion(self.plot.data,
                                                   self.plot.image))
        self.transect.setData(trans)
        # self.transect.setData(trans[0], num.linspace(0, 100, trans[0].size))


class QKiteToolHistogram(pg.HistogramLUTWidget):
    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)
        self.plot = plot

        _zero_marker = pg.InfiniteLine(pos=0, angle=0, pen='w', movable=False)
        _zero_marker.setValue(0.)
        _zero_marker.setZValue(1000)
        self.vb.addItem(_zero_marker)

        self.symetricColormap()
        # self.isoCurveControl()

    def symetricColormap(self):
        default_cmap = {'ticks':
                        [[0., (106, 0, 31, 255)],
                         [.5, (255, 255, 255, 255)],
                         [1., (8, 54, 104, 255)]],
                        'mode': 'rgb'}
        lvl_min = num.nanmin(self.plot.data)
        lvl_max = num.nanmax(self.plot.data)
        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(default_cmap)
        self.setLevels(-abs_range, abs_range)

    def isoCurveControl(self):
        iso_ctrl = pg.InfiniteLine(pos=0, angle=0, pen='g', movable=True)
        iso_ctrl.setValue(0.)
        iso_ctrl.setZValue(1000)

        def isolineChange():
            self.plot.iso.setLevel(iso_ctrl.value())

        iso_ctrl.sigDragged.connect(isolineChange)
        self.vb.addItem(iso_ctrl)


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
            # 'Colormap': QKiteToolGradient,
            'Transect': QKiteToolTransect,
            'Histogram': QKiteToolHistogram,
        }

        QKiteDock.__init__(self, scene)


class QKiteQuadtreeDock(QKiteDock):
    def __init__(self, quadtree):
        self.title = 'Quadtree'
        self.main_widget = QKiteQuadtreePlot
        self.tools = {
            'Quadtree Control': QKiteToolQuadtree,
            'Components': QKiteToolComponents,
            'Histogram': QKiteToolHistogram,
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
