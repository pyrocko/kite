#!/usr/bin/python2
from PySide import QtGui
from PySide import QtCore
from qt_utils import QDoubleSlider
from .tab import QKiteDock, QKiteToolComponents, QKitePlot

import pyqtgraph as pg


class QKiteQuadtreeDock(QKiteDock):
    def __init__(self, quadtree):
        self.title = 'Quadtree'
        self.main_widget = QKiteQuadtreePlot
        self.tools = {
            'Quadtree Control': QKiteToolQuadtree,
            'Components': QKiteToolComponents,
            # 'Histogram': QKiteToolHistogram,
        }

        QKiteDock.__init__(self, quadtree)


class QKiteQuadtreePlot(QKitePlot):
    def __init__(self, quadtree):

        self.components_available = {
            'mean': ['Mean Displacement',
                     lambda qt: qt.leaf_matrix_means],
            'median': ['Median Displacement',
                       lambda qt: qt.leaf_matrix_medians],
            'weight': ['Absolute Weight',
                       lambda qt: qt.leaf_matrix_weights],
        }
        self._component = 'median'

        QKitePlot.__init__(self, container=quadtree)
        self.quadtree = self.container

        focalpoint_color = (255, 23, 68, 255)
        focalpoint_outline_color = (255, 255, 255, 200)
        self.focal_points = pg.ScatterPlotItem(
                                size=3.,
                                pen=pg.mkPen(focalpoint_outline_color,
                                             width=.5),
                                brush=pg.mkBrush(focalpoint_color))

        self.addItem(self.focal_points)
        self.updateFocalPoints()

        self.quadtree.treeUpdate.subscribe(self.update)
        self.quadtree.treeUpdate.subscribe(self.updateFocalPoints)

    def updateFocalPoints(self):
        self.focal_points.setData(pos=self.quadtree.leaf_focal_points,
                                  pxMode=True)


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.quadtree = plot.container

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getEpsilonChanger())
        self.layout.addWidget(self.getNaNFractionChanger())
        self.layout.addWidget(self.getTileSizeChanger())
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

        group = QtGui.QGroupBox('Epsilon')
        group.setToolTip('''<p>Standard deviation/split
                        method of each tile is >= epsilon</p>''')
        group.setLayout(layout)

        return group

    def getNaNFractionChanger(self):
        layout = QtGui.QHBoxLayout()

        slider = QDoubleSlider(QtCore.Qt.Horizontal)
        spin = QtGui.QDoubleSpinBox()

        def changeNaNFraction():
            nan_allowed = round(spin.value(), 3)
            self.quadtree.nan_allowed = nan_allowed
            slider.setValue(nan_allowed)

        for wdgt in [slider, spin]:
            wdgt.setValue(self.quadtree.nan_allowed or 1.)
            wdgt.setRange(0., 1.)
            wdgt.setSingleStep(.05)

        spin.setDecimals(2)

        spin.valueChanged.connect(changeNaNFraction)
        slider.valueChanged.connect(lambda: spin.setValue(round(slider.value(),
                                                                3)))

        layout.addWidget(spin)
        layout.addWidget(slider)

        group = QtGui.QGroupBox('Allowed NaN Fraction')
        group.setToolTip('''<p>Maximum <b>NaN pixel
            fraction allowed</b> per tile</p>''')
        group.setLayout(layout)

        return group

    def getTileSizeChanger(self):
        layout = QtGui.QGridLayout()

        slider_smin = QtGui.QSlider(QtCore.Qt.Horizontal)
        spin_smin = QtGui.QSpinBox()
        slider_smax = QtGui.QSlider(QtCore.Qt.Horizontal)
        spin_smax = QtGui.QSpinBox()

        def changeTileLimits():
            smin, smax = spin_smin.value(), spin_smax.value()
            if smax == spin_smax.maximum() or smax == 0.:
                smax = -9999.

            self.quadtree.tile_size_lim = (smin, smax)
            slider_smin.setValue(spin_smin.value())
            slider_smax.setValue(spin_smax.value())

        for wdgt in [slider_smax, slider_smin, spin_smax, spin_smin]:
            wdgt.setRange(0, 25000)
            wdgt.setSingleStep(50)

        for wdgt in [slider_smin, spin_smin]:
            wdgt.setValue(self.quadtree.tile_size_lim[0])
        slider_smin.valueChanged.connect(
            lambda: spin_smin.setValue(slider_smin.value()))
        spin_smin.valueChanged.connect(changeTileLimits)
        spin_smin.setSuffix(' m')

        for wdgt in [slider_smax, spin_smax]:
            wdgt.setValue(self.quadtree.tile_size_lim[1])
        slider_smax.valueChanged.connect(
            lambda: spin_smax.setValue(slider_smax.value()))
        spin_smax.valueChanged.connect(changeTileLimits)
        spin_smax.setSpecialValueText('inf')
        spin_smax.setSuffix(' m')

        layout.addWidget(QtGui.QLabel('Min',
                                      toolTip='Minimum tile size in meter'),
                         1, 1)
        layout.addWidget(spin_smin, 1, 2)
        layout.addWidget(slider_smin, 1, 3)

        layout.addWidget(QtGui.QLabel('Max',
                                      toolTip='Maximum tile size in meter'),
                         2, 1)
        layout.addWidget(spin_smax, 2, 2)
        layout.addWidget(slider_smax, 2, 3)

        group = QtGui.QGroupBox('Tile Size Limits')
        group.setToolTip('''<p>Tile size limits
                         overwrite the desired epsilon parameter</p>''')
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
            btn.setChecked(method == self.quadtree.config.split_method)
            btn.clicked.connect(partial(changeMethod, method))

            layout.addWidget(btn)

        group = QtGui.QGroupBox('Tile Split Criteria')
        group.setLayout(layout)

        return group

    def getInfoPanel(self):
        layout = QtGui.QVBoxLayout()
        info_text = QtGui.QLabel()

        def updateInfoText():
            infos = [
                ('Leaf Count', '<b>%d</b>' % len(self.quadtree.leafs)),
                ('Epsilon current', '%0.3f' % self.quadtree.epsilon),
                ('Epsilon limit', '%0.3f' % self.quadtree._epsilon_limit),
                ('Allowed NaN fraction',
                    '%d%%' % int(self.quadtree.nan_allowed * 100)
                    if self.quadtree.nan_allowed != -9999. else 'inf'),
                ('Min tile size', '%d m' % self.quadtree.tile_size_lim[0]),
                ('Max tile size', '%d m' % self.quadtree.tile_size_lim[1]
                    if self.quadtree.tile_size_lim[1] != -9999. else 'inf'),
            ]
            _text = '<table>'
            for (param, value) in infos:
                _text += '''<tr><td style='padding-right: 10px'>%s:</td>
                    <td><b>%s</td></tr>''' % (param, value)
            _text += '</table>'
            info_text.setText(_text)

        updateInfoText()
        self.quadtree.treeUpdate.subscribe(updateInfoText)

        layout.addWidget(info_text)
        group = QtGui.QGroupBox('Quadtree Information')
        group.setLayout(layout)

        return group
