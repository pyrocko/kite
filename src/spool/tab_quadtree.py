#!/usr/bin/python2
from __future__ import division, absolute_import, print_function, \
    unicode_literals
from PySide import QtGui
from PySide import QtCore
from .utils_qt import QDoubleSlider
from .common import QKiteView, QKitePlot, QKiteParameterGroup

import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes


class QKiteQuadtree(QKiteView):
    def __init__(self, spool):
        quadtree = spool.scene.quadtree
        self.title = 'Scene.quadtree'
        self.main_widget = QKiteQuadtreePlot(quadtree)
        self.tools = {
            # 'Quadtree Parameters': QKiteToolQuadtree(quadtree),
        }

        self.parameters = [
            QKiteParamQuadtree(spool, self.main_widget, expanded=True)
        ]

        QKiteView.__init__(self)


class QKiteQuadtreePlot(QKitePlot):
    def __init__(self, quadtree):

        self.components_available = {
            'mean': ['Node.mean displacement',
                     lambda qt: qt.leaf_matrix_means],
            'median': ['Node.median displacement',
                       lambda qt: qt.leaf_matrix_medians],
            'weight': ['Node.weight covariance',
                       lambda qt: qt.leaf_matrix_weights],
        }
        self._component = 'median'

        QKitePlot.__init__(self, container=quadtree)
        self.quadtree = self.container

        # http://paletton.com
        focalpoint_color = (45, 136, 45)
        # focalpoint_outline_color = (255, 255, 255, 200)
        focalpoint_outline_color = (3, 212, 3)
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


class QKiteParamQuadtree(QKiteParameterGroup):
    def __init__(self, spool, plot, *args, **kwargs):
        self.quadtree = spool.scene.quadtree
        self.plot = plot
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.quadtree'
        self.parameters = ['nleafs', 'nnodes', 'epsilon_limit']

        QKiteParameterGroup.__init__(self, self.quadtree, **kwargs)
        self.quadtree.treeUpdate.subscribe(self.updateValues)

        # Epsilon control
        def updateEpsilon():
            self.quadtree.epsilon = self.epsilon.value()

        p = {'name': 'epsilon',
             'value': self.quadtree.epsilon,
             'type': 'float',
             'min': self.quadtree.epsilon_limit,
             'max': 3 * self.quadtree.epsilon,
             'step': round((self.quadtree.epsilon -
                            self.quadtree.epsilon_limit)*.2, 3),
             'editable': True}
        self.epsilon = pTypes.SimpleParameter(**p)
        self.epsilon.sigValueChanged.connect(updateEpsilon)

        # Epsilon control
        def updateNanFrac():
            self.quadtree.nan_allowed = self.nan_allowed.value()

        p = {'name': 'nan_allowed',
             'value': self.quadtree.nan_allowed,
             'type': 'float',
             'min': 0.,
             'max': 1.,
             'step': 0.05,
             'editable': True}
        self.nan_allowed = pTypes.SimpleParameter(**p)
        self.nan_allowed.sigValueChanged.connect(updateNanFrac)

        # Tile size controls
        def updateTileSizeMin():
            self.quadtree.tile_size_min = self.tile_size_min.value()

        p = {'name': 'tile_size_min',
             'value': self.quadtree.tile_size_min,
             'type': 'int',
             'min': 100,
             'max': 50000,
             'step': 250,
             'editable': True}
        self.tile_size_min = pTypes.SimpleParameter(**p)

        def updateTileSizeMax():
            self.quadtree.tile_size_max = self.tile_size_max.value()

        p.update({'name': 'tile_size_max',
                  'value': self.quadtree.tile_size_max})
        self.tile_size_max = pTypes.SimpleParameter(**p)
        self.tile_size_min.sigValueChanged.connect(updateTileSizeMin)
        self.tile_size_max.sigValueChanged.connect(updateTileSizeMax)

        # Component control
        def changeComponent():
            self.plot.component = self.components.value()

        p = {'name': 'display',
             'values': {
                'QuadNode.mean': 'mean',
                'QuadNode.median': 'median',
                'QuadNode.weight': 'weight',
             },
             'value': 'mean'}
        self.components = pTypes.ListParameter(**p)
        self.components.sigValueChanged.connect(changeComponent)

        def changeSplitMethod():
            self.quadtree.setSplitMethod(self.split_method.value())

        p = {'name': 'setSplitMethod',
             'values': {
                'Mean Std (Sigurjonson, 2001)': 'mean_std',
                'Median Std (Sigurjonson, 2001)': 'median_std',
                'Std (Sigurjonson, 2001)': 'std',
             },
             'value': 'mean'}
        self.split_method = pTypes.ListParameter(**p)
        self.split_method.sigValueChanged.connect(changeSplitMethod)

        self.pushChild(self.split_method)
        self.pushChild(self.tile_size_max)
        self.pushChild(self.tile_size_min)
        self.pushChild(self.nan_allowed)
        self.pushChild(self.epsilon)
        self.pushChild(self.components)


class QKiteToolQuadtree(QtGui.QWidget):
    def __init__(self, quadtree):
        QtGui.QWidget.__init__(self)
        self.quadtree = quadtree

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
                wdgt.setRange(self.quadtree.epsilon_limit,
                              3*self.quadtree.epsilon)
                wdgt.setSingleStep(round((self.quadtree.epsilon -
                                          self.quadtree.epsilon_limit)*.2, 3))

        spin.setDecimals(3)
        updateRange()

        self.quadtree.splitMethodChanged.subscribe(updateRange)
        spin.valueChanged.connect(changeEpsilon)
        slider.valueChanged.connect(lambda: spin.setValue(round(slider.value(),
                                                                3)))

        layout.addWidget(spin)
        layout.addWidget(slider)

        group = QtGui.QGroupBox('Scene.quadtree.epsilon')
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

        group = QtGui.QGroupBox('Scene.quadtree.nan_allowed (NaN as fraction)')
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

            self.quadtree.tile_size_min = smin
            self.quadtree.tile_size_max = smax

            slider_smin.setValue(spin_smin.value())
            slider_smax.setValue(spin_smax.value())

        for wdgt in [slider_smax, slider_smin, spin_smax, spin_smin]:
            wdgt.setRange(0, 25000)
            wdgt.setSingleStep(50)

        for wdgt in [slider_smin, spin_smin]:
            wdgt.setValue(self.quadtree.tile_size_min)
        slider_smin.valueChanged.connect(
            lambda: spin_smin.setValue(slider_smin.value()))
        spin_smin.valueChanged.connect(changeTileLimits)
        spin_smin.setSuffix(' m')

        for wdgt in [slider_smax, spin_smax]:
            wdgt.setValue(self.quadtree.tile_size_max)
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

        group = QtGui.QGroupBox('Scene.quadtree.tile_size_lim')
        group.setToolTip('<p>Tile size limits, '
                         'overwrites the desired epsilon parameter</p>')
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

        group = QtGui.QGroupBox('Scene.quadtree.setSplitMethod')
        group.setLayout(layout)

        return group

    def getInfoPanel(self):
        layout = QtGui.QVBoxLayout()
        info_text = QtGui.QLabel()

        def updateInfoText():
            infos = [
                ('Leaf Count', '<b>%d</b>' % len(self.quadtree.leafs)),
                ('Epsilon current', '%0.3f' % self.quadtree.epsilon),
                ('Epsilon limit', '%0.3f' % self.quadtree.epsilon_limit),
                ('Allowed NaN fraction',
                    '%d%%' % int(self.quadtree.nan_allowed * 100)
                    if self.quadtree.nan_allowed != -9999. else 'inf'),
                ('Min tile size', '%d m' % self.quadtree.tile_size_min),
                ('Max tile size', '%d m' % self.quadtree.tile_size_max),
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
