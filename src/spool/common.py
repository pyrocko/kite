#!/usr/bin/python2
from __future__ import division, absolute_import, print_function, \
    unicode_literals

from PySide import QtCore
import time
import numpy as num
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph import dockarea
from .utils_qt import _viridis_data
from ..meta import calcPrecission

__all__ = ['QKiteView', 'QKitePlot', 'QKiteToolColormap',
           'QKiteParameterGroup']


class QKiteView(dockarea.DockArea):
    def __init__(self):
        dockarea.DockArea.__init__(self)
        self.tool_docks = []

        dock_main =\
            dockarea.Dock(self.title,
                          autoOrientation=False,
                          widget=self.main_widget)
        dock_colormap =\
            dockarea.Dock('Colormap',
                          autoOrientation=False,
                          widget=QKiteToolColormap(self.main_widget))
        dock_colormap.setStretch(1, None)

        for i, (name, tool) in enumerate(self.tools.iteritems()):
            self.tool_docks.append(
                dockarea.Dock(name,
                              widget=tool,
                              size=(2, 2),
                              autoOrientation=False))
            self.addDock(self.tool_docks[-1], position='bottom')

        self.addDock(dock_main, position='left')
        self.addDock(dock_colormap, position='right')


class QKitePlot(pg.PlotWidget):
    def __init__(self, scene_proxy):
        pg.PlotWidget.__init__(self)
        self.scene_proxy = scene_proxy
        self.draw_time = 0.

        border_pen = pg.mkPen(255, 255, 255, 50)
        self.image = pg.ImageItem(None,
                                  autoDownsample=False,
                                  border=border_pen,
                                  useOpenGL=True)

        self.setAspectLocked(True)
        self.plotItem.getAxis('left').setZValue(100)
        self.plotItem.getAxis('bottom').setZValue(100)
        self.setLabels(bottom={'East', 'm'},
                       left={'North', 'm'},)

        self.hint = {
            'east': 0.,
            'north': 0.,
            'value': num.nan,
            'measure': self.component.title(),
            'vlength': '03',
            'precision': '3',
        }

        self.hint_text = pg.LabelItem(text='',
                                      justify='right', size='8pt',
                                      parent=self.plotItem)
        self.hint_text.anchor(itemPos=(1., 0.), parentPos=(1., 0.))
        self.hint_text.template =\
            '<span style="font-family: monospace; color: #fff;'\
            'background-color: #000;">'\
            'East {east:08.2f} m | North {north:08.2f} m | '\
            '{measure} {value:{length}.{precision}f}</span>'

        self.addItem(self.image)
        self.update()

        self.transFromFrame()
        self._move_sig = pg.SignalProxy(self.image.scene().sigMouseMoved,
                                        rateLimit=25, slot=self.mouseMoved)
        # self.addIsocurve()
        # self.scalebar()

    def transFromFrame(self):
        self.image.resetTransform()
        self.image.scale(self.scene_proxy.frame.dE, self.scene_proxy.frame.dN)

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
        _data = self.components_available[self.component][1](self.scene_proxy)
        self.hint['precision'], self.hint['vlength'] = calcPrecission(_data)
        return _data  # num.nan_to_num(_data)

    @QtCore.Slot()
    def update(self):
        ts = time.time()
        self.image.updateImage(self.data.T)
        self.draw_time = time.time() - ts
        self.mouseMoved()
        # self.addIsocurves()

    def addIsocurve(self, level=0.):
        iso = pg.IsocurveItem(level=level, pen='g')
        iso.setZValue(1000)
        iso.setData(pg.gaussianFilter(self.data, (5, 5)))
        iso.setParentItem(self.image)

        self.iso = iso

    @QtCore.Slot(object)
    def mouseMoved(self, event=None):
        if event is None:
            pass
        elif self.image.sceneBoundingRect().contains(event[0]):
            map_pos = self.plotItem.vb.mapSceneToView(event[0])
            if not map_pos.isNull():
                img_pos = self.image.mapFromScene(event).data
                value = self.image.image[int(img_pos().x()),
                                         int(img_pos().y())]

                self.hint['east'] = map_pos.y()
                self.hint['north'] = map_pos.x()
                self.hint['value'] = value
        self.hint['length'] = '03' if num.isnan(self.hint['value'])\
                              else self.hint['vlength']
        self.hint_text.setText(self.hint_text.template.format(**self.hint))


class QKiteToolColormap(pg.HistogramLUTWidget):
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
        self._plot.image.sigImageChanged.connect(self.imageChanged)
        # self.isoCurveControl()

    def imageChanged(self):
        if self._plot.component == 'weight':
            self.setQualitativeColormap()
        else:
            self.setSymColormap()

    def setSymColormap(self):
        cmap = {'ticks':
                [[0., (0, 0, 0, 255)],
                 [1e-3, (106, 0, 31, 255)],
                 [.5, (255, 255, 255, 255)],
                 [1., (8, 54, 104, 255)]],
                'mode': 'rgb'}
        cmap = {'ticks':
                [[0., (0, 0, 0)],
                 [1e-3, (172, 56, 56)],
                 [.5, (255, 255, 255)],
                 [1., (51, 53, 120)]],
                'mode': 'rgb'}
        lvl_min = num.nanmin(self._plot.data)
        lvl_max = num.nanmax(self._plot.data)
        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(cmap)
        self.setLevels(-abs_range, abs_range)

    def setQualitativeColormap(self):
        l = len(_viridis_data) - 1
        cmap = {'mode': 'rgb'}
        cmap['ticks'] = [[float(i)/l, c] for i, c in enumerate(_viridis_data)]
        self.gradient.restoreState(cmap)
        self.setLevels(num.nanmin(self._plot.data),
                       num.nanmax(self._plot.data))

    def isoCurveControl(self):
        iso_ctrl = pg.InfiniteLine(pos=0, angle=0, pen='g', movable=True)
        iso_ctrl.setValue(0.)
        iso_ctrl.setZValue(1000)

        def isolineChange():
            self._plot.iso.setLevel(iso_ctrl.value())

        iso_ctrl.sigDragged.connect(isolineChange)
        self.vb.addItem(iso_ctrl)


class QKiteParameterGroup(pTypes.GroupParameter):
    def __init__(self, model, model_attr=None, **kwargs):
        self.model = model
        self.model_attr = model_attr

        if isinstance(self.parameters, list):
            self.parameters = dict.fromkeys(self.parameters)

        pTypes.GroupParameter.__init__(self, **kwargs)
        self.updateValues()

    def updateValues(self):
        if not hasattr(self, 'parameters'):
            return

        if self.model_attr is not None and\
           hasattr(self.model, self.model_attr):
            model = getattr(self.model, self.model_attr)
        else:
            model = self.model

        for param, f in self.parameters.iteritems():
            QtCore.QCoreApplication.processEvents()
            try:
                if callable(f):
                    value = f(model)
                    # print('Updating %s: %s (func)' % (param, value))
                else:
                    value = getattr(model, param)
                    # print('Updating %s: %s (getattr)' % (param, value))
            except AttributeError:
                value = 'n/a'
            try:
                self.child(param).setValue(value)
            except Exception:
                self.addChild({'name': param,
                               'value': value,
                               'type': 'str',
                               'readonly': True})

    def pushChild(self, child, **kwargs):
        self.insertChild(0, child, **kwargs)


class QKiteThread(QtCore.QThread):
    def __init__(self, work=None, args=(), kwargs={}, parent=None):
        self.work = work
        self.args = args
        self.kwargs = kwargs
        print(args)
        print(kwargs)
        QtCore.QThread.__init__(self, parent)

    def run(self):
        if callable(self.work):
            self.work(*self.args, **self.kwargs)
        else:
            raise TypeError('worker must be callable!')
