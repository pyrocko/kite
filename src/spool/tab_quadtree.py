#!/usr/bin/python2
from __future__ import division, absolute_import, print_function, \
    unicode_literals
from .utils_qt import SliderWidgetParameterItem
from .common import QKiteView, QKitePlot, QKiteParameterGroup

import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes


class QKiteQuadtree(QKiteView):
    def __init__(self, spool):
        quadtree = spool.scene.quadtree
        self.title = 'Scene.quadtree'
        self.main_widget = QKiteQuadtreePlot(quadtree)
        self.tools = {}

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
        self.quadtree = quadtree

        QKitePlot.__init__(self, container=quadtree)

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

        self.quadtree.evConfigUpdated.subscribe(self.update)
        self.quadtree.evConfigUpdated.subscribe(self.updateFocalPoints)

    def updateFocalPoints(self):
        if self.quadtree.leaf_focal_points.size == 0:
            self.focal_points.clear()
        else:
            self.focal_points.setData(pos=self.quadtree.leaf_focal_points,
                                      pxMode=True)


class QKiteParamQuadtree(QKiteParameterGroup):
    def __init__(self, spool, plot, *args, **kwargs):
        self.quadtree = spool.scene.quadtree
        self.plot = plot
        self.sig_guard = True

        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.quadtree'
        self.parameters = ['nleafs', 'nnodes', 'epsilon_limit']

        QKiteParameterGroup.__init__(self, self.quadtree, **kwargs)
        self.quadtree.evConfigUpdated.subscribe(self.onConfigUpdate)

        def updateGuard(func):
            def wrapper(*args, **kwargs):
                if not self.sig_guard:
                    func()
            return wrapper

        # Epsilon control
        @updateGuard
        def updateEpsilon():
            self.quadtree.epsilon = self.epsilon.value()

        p = {'name': 'epsilon',
             'value': self.quadtree.epsilon,
             'type': 'float',
             'step': round((self.quadtree.epsilon -
                            self.quadtree.epsilon_limit)*.1, 3),
             'limits': (self.quadtree.epsilon_limit,
                        2*self.quadtree.epsilon),
             'editable': True}
        self.epsilon = pTypes.SimpleParameter(**p)
        self.epsilon.itemClass = SliderWidgetParameterItem
        self.epsilon.sigValueChanged.connect(updateEpsilon)

        # Epsilon control
        @updateGuard
        def updateNanFrac():
            self.quadtree.nan_allowed = self.nan_allowed.value()

        p = {'name': 'nan_allowed',
             'value': self.quadtree.nan_allowed,
             'type': 'float',
             'step': 0.05,
             'limits': (0., 1.),
             'editable': True, }
        self.nan_allowed = pTypes.SimpleParameter(**p)
        self.nan_allowed.itemClass = SliderWidgetParameterItem
        self.nan_allowed.sigValueChanged.connect(updateNanFrac)

        # Tile size controls
        @updateGuard
        def updateTileSizeMin():
            self.quadtree.tile_size_min = self.tile_size_min.value()

        p = {'name': 'tile_size_min',
             'value': self.quadtree.tile_size_min,
             'type': 'int',
             'limits': (50, 50000),
             'step': 100,
             'editable': True,
             'siPrefix': 'm'}
        self.tile_size_min = pTypes.SimpleParameter(**p)
        self.tile_size_min.itemClass = SliderWidgetParameterItem

        @updateGuard
        def updateTileSizeMax():
            self.quadtree.tile_size_max = self.tile_size_max.value()

        p.update({'name': 'tile_size_max',
                  'value': self.quadtree.tile_size_max})
        self.tile_size_max = pTypes.SimpleParameter(**p)
        self.tile_size_max.itemClass = SliderWidgetParameterItem

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

        self.sig_guard = False
        self.pushChild(self.split_method)
        self.pushChild(self.tile_size_max)
        self.pushChild(self.tile_size_min)
        self.pushChild(self.nan_allowed)
        self.pushChild(self.epsilon)
        self.pushChild(self.components)

    def onConfigUpdate(self):
        self.sig_guard = True
        for p in ['epsilon', 'nan_allowed',
                  'tile_size_min', 'tile_size_max']:
            param = getattr(self, p)
            param.setValue(getattr(self.quadtree, p))
        self.sig_guard = False
