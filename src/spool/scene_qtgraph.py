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
        # self.image.setAutoDownsample(True)

        self.setAspectLocked(True)
        self.plotItem.getAxis('left').setZValue(100)
        self.plotItem.getAxis('bottom').setZValue(100)
        # self.setBackground((255, 255, 255, 255))

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
        _data = self.components_available[self.component][1](self.container)
        return _data  # num.nan_to_num(_data)

    def update(self):
        self.image.updateImage(self.data, autoDownsample=True)
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
            # 'displacement': {
            #     'button'
            #     'func'
            #     'detailed desc'
            #     'colormap_autoscale'
            # }
            'displacement': ['LOS Displacement', lambda sc: sc.displacement],
            'theta': ['LOS Theta', lambda sc: sc.theta],
            'phi': ['LOS Phi', lambda sc: sc.phi],
            'thetaDeg': ['LOS Theta degree', lambda sc: sc.los.degTheta],
            'phiDeg': ['LOS Phi degree', lambda sc: sc.los.degPhi],
            'unitE': ['LOS unitE', lambda sc: sc.los.unitE],
            'unitN': ['LOS unitN', lambda sc: sc.los.unitN],
            'unitU': ['LOS unitU', lambda sc: sc.los.unitU],
        }
        self._component = 'displacement'

        _QKitePlot.__init__(self, container=scene)
        # self.transformToUTM()

    def transformToUTM(self):
        padding = 100
        ll_x, ll_y, ur_x, ur_y, dx, dy = self.container.getUTMExtend()
        print self.container.getUTMExtend()

        self.image.translate(ll_x, ll_y)
        self.image.scale(dx, dy)
        self.setLimits(xMin=ll_x-dx*padding,
                       xMax=ur_x+dx*padding,
                       yMin=ll_y-dy*padding,
                       yMax=ur_y+dy*padding)


class QKiteQuadtreePlot(_QKitePlot):
    def __init__(self, quadtree):

        self.components_available = {
            'mean': ['Mean Displacement',
                     lambda qt: qt.leaf_matrix_means],
            'median': ['Median Displacement',
                       lambda qt: qt.leaf_matrix_medians],
        }
        self._component = 'median'

        _QKitePlot.__init__(self, container=quadtree)
        self.quadtree = self.container

        _focalp_color = (0, 0, 0, 100)
        self.focal_points = pg.ScatterPlotItem(size=1.5,
                                               pen=pg.mkPen(_focalp_color,
                                                            width=1),
                                               brush=pg.mkBrush(_focalp_color))

        self.addItem(self.focal_points)
        self.updateFocalPoints()

        self.container.subscribe(self.update)
        self.container.subscribe(self.updateFocalPoints)

    def updateFocalPoints(self):
        self.focal_points.setData(pos=self.container.leaf_focal_points)


class QKiteToolComponents(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getComponentsGroup())
        self.layout.addWidget(self.getInfoPanel())
        self.layout.addStretch(3)

    def getComponentsGroup(self):
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

        group = QtGui.QGroupBox('Scene Components')
        group.setLayout(layout)

        return group

    def getInfoPanel(self):
        layout = QtGui.QVBoxLayout()
        info_text = QtGui.QLabel()

        table_content = [
            ['Component', '<b>%s</b>' %
             self.plot.components_available[self.plot.component][0]],
            ['Max value', '%0.4f' % num.nanmax(self.plot.data)],
            ['Min value', '%0.4f' % num.nanmin(self.plot.data)],
            ['Mean value', '%0.4f' % num.nanmean(self.plot.data)],
            ['Resolution px', '%d x %d' % (self.plot.data.shape[0],
                                           self.plot.data.shape[1])],
        ]

        def updateInfoText():
            rstr = '<table>'
            for (metric, value) in table_content:
                rstr += '<tr><td style="padding-right: 15px">%s:</td>' \
                        '<td>%s</td></tr>' % (metric, value)
            rstr += '</table>'

            info_text.setText(rstr)

        updateInfoText()
        self.plot.image.sigImageChanged.connect(updateInfoText)

        layout.addWidget(info_text)
        group = QtGui.QGroupBox('Component Information')
        group.setLayout(layout)

        return group


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.quadtree = plot.container

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getEpsilonChanger())
        self.layout.addWidget(self.getNaNFractionChanger())
        self.layout.addWidget(self.getMethodsChanger())
        self.layout.addWidget(self.getInfoPanel())
        self.layout.addStretch(3)

    def getEpsilonChanger(self):
        layout = QtGui.QHBoxLayout()

        slider = QDoubleSlider(QtCore.Qt.Horizontal)
        spin = QtGui.QDoubleSpinBox()

        def changeEpsilon():
            epsilon = round(spin.value(), 3)
            self.quadtree.epsilon = epsilon
            slider.setValue(epsilon)

        def updateRange():
            for wdgt in [slider, spin]:
                wdgt.setValue(self.quadtree.epsilon)
                wdgt.setRange(self.quadtree._epsilon_limit,
                              3*self.quadtree.epsilon)
                wdgt.setSingleStep(round((self.quadtree.epsilon -
                                          self.quadtree._epsilon_limit)*.2, 3))

        spin.setDecimals(3)
        updateRange()

        self.quadtree.splitMethodChanged.subscribe(updateRange)
        spin.valueChanged.connect(changeEpsilon)
        slider.valueChanged.connect(lambda: spin.setValue(round(slider.value(),
                                                                3)))

        layout.addWidget(spin)
        layout.addWidget(slider)

        group = QtGui.QGroupBox('Epsilon Control')
        group.setToolTip('''<p>Standard deviation/split
                        method of each tile is >= epsilon</p>''')
        group.setLayout(layout)

        return group

    def getNaNFractionChanger(self):
        layout = QtGui.QHBoxLayout()

        slider = QDoubleSlider(QtCore.Qt.Horizontal)
        spin = QtGui.QDoubleSpinBox()

        def changeNaNFraction():
            max_nan = round(spin.value(), 3)
            self.quadtree.max_nan = max_nan
            slider.setValue(max_nan)

        def updateRange():
            for wdgt in [slider, spin]:
                wdgt.setValue(self.quadtree.max_nan or 1.)
                wdgt.setRange(0., 1.)
                wdgt.setSingleStep(.05)

        spin.setDecimals(2)
        updateRange()

        spin.valueChanged.connect(changeNaNFraction)
        slider.valueChanged.connect(lambda: spin.setValue(round(slider.value(),
                                                                3)))

        layout.addWidget(spin)
        layout.addWidget(slider)

        group = QtGui.QGroupBox('Allowed NaN Fraction')
        group.setToolTip('''<p>Maximum NaN pixel fraction per tile</p>''')
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

        self.trans_plot = pg.PlotDataItem(antialias=True)
        self.addItem(self.trans_plot)
        self.plotItem.getAxis('bottom').setLabel('Distance / m')
        self.plotItem.getAxis('left').setLabel('Displacement / m')
        self.addPolyLine()
        self.polyline.sigRegionChangeFinished.connect(self.updateTransPlot)
        self.plot.image.sigImageChanged.connect(self.updateTransPlot)
        # self.plot.image.sigImageChanged.connect(self.addPolyLine)

    def addPolyLine(self):
        [[xmin, xmax], [ymin, ymax]] = self.plot.viewRange()
        self.polyline = pg.PolyLineROI(positions=[(xmin+(xmax-xmin)*.4,
                                                   ymin+(ymax-ymin)*.4),
                                                  (xmin+(xmax-xmin)*.6,
                                                   ymin+(ymax-ymin)*.6)],
                                       pen=pg.mkPen('g', width=2))
        self.plot.addItem(self.polyline)

    def updateTransPlot(self):
        trans = num.ndarray((0))
        length = 0
        for line in self.polyline.segments:
            trans = num.append(trans, line.getArrayRegion(self.plot.data,
                                                          self.plot.image))
            p1, p2 = line.listPoints()
            length += (p2-p1).length()
        # interpolate over NaNs
        nans, x = num.isnan(trans), lambda z: z.nonzero()[0]
        trans[nans] = num.interp(x(nans), x(~nans), trans[~nans])

        self.trans_plot.setData(num.linspace(0, length, trans.size), trans)


class QKiteToolHistogram(pg.HistogramLUTWidget):
    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)

        self._plot = plot

        _zero_marker = pg.InfiniteLine(pos=0, angle=0, pen='w', movable=False)
        _zero_marker.setValue(0.)
        _zero_marker.setZValue(1000)
        self.vb.addItem(_zero_marker)

        self.axis.setLabel('Displacement / m')
        # self.plot.rotate(-90)
        # self.layout.rotate(90)
        # self.gradient.setOrientation('bottom')
        self.setSymColormap()
        self._plot.image.sigImageChanged.connect(self.setSymColormap)
        # self.isoCurveControl()

    def setSymColormap(self):
        default_cmap = {'ticks':
                        [[0., (0, 0, 0, 255)],
                         [1e-3, (106, 0, 31, 255)],
                         [.5, (255, 255, 255, 255)],
                         [1., (8, 54, 104, 255)]],
                        'mode': 'rgb'}
        lvl_min = num.nanmin(self._plot.data)
        lvl_max = num.nanmax(self._plot.data)
        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(default_cmap)
        self.setLevels(-abs_range, abs_range)

    def isoCurveControl(self):
        iso_ctrl = pg.InfiniteLine(pos=0, angle=0, pen='g', movable=True)
        iso_ctrl.setValue(0.)
        iso_ctrl.setZValue(1000)

        def isolineChange():
            self._plot.iso.setLevel(iso_ctrl.value())

        iso_ctrl.sigDragged.connect(isolineChange)
        self.vb.addItem(iso_ctrl)


class QKiteDock(dockarea.DockArea):
    def __init__(self, container):
        dockarea.DockArea.__init__(self)

        main_dock = dockarea.Dock(self.title)
        main_widget = self.main_widget(container)
        main_dock.addWidget(main_widget)

        for i, (name, tool) in enumerate(self.tools.iteritems()):
            tool_dock = dockarea.Dock(name, widget=tool(main_widget),
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
