#!/usr/bin/python2
import pyqtgraph as pg
import numpy as num
import pyqtgraph.parametertree.parameterTypes as pTypes

from .common import QKiteView, QKitePlot, QKiteParameterGroup
from .utils_qt import loadUi
from collections import OrderedDict
from PySide import QtGui
from os import path

__all__ = ['QKiteScene']


class QKiteScene(QKiteView):
    def __init__(self, spool):
        scene = spool.scene

        self.title = 'Scene'
        self.main_widget = QKiteScenePlot(scene)
        self.tools = {
            # 'Components': QKiteToolComponents(self.main_widget),
            # 'Displacement Transect': QKiteToolTransect(self.main_widget),
        }

        self.parameters = [QKiteParamScene(spool, self.main_widget)]
        self.parameters[-1].addChild(
            QKiteParamSceneFrame(spool, expanded=False))
        self.parameters[-1].addChild(
            QKiteParamSceneMeta(spool, expanded=False))

        self.dialogTransect = QKiteToolTransect(self.main_widget)

        spool.actionTransect.triggered.connect(self.dialogTransect.show)
        spool.actionTransect.setEnabled(True)

        QKiteView.__init__(self)


class QKiteScenePlot(QKitePlot):
    def __init__(self, scene):

        self.components_available = {
            'displacement': ['Scene.displacement', lambda sc: sc.displacement],
            'theta': ['Scene.theta', lambda sc: sc.theta],
            'phi': ['Scene.phi', lambda sc: sc.phi],
            'thetaDeg': ['Scene.thetaDeg', lambda sc: sc.thetaDeg],
            'phiDeg': ['Scene.phiDeg', lambda sc: sc.phiDeg],
            'unitE': ['Scene.los.unitE', lambda sc: sc.los.unitE],
            'unitN': ['Scene.los.unitN', lambda sc: sc.los.unitN],
            'unitU': ['Scene.los.unitU', lambda sc: sc.los.unitU],
        }
        self._component = 'displacement'

        QKitePlot.__init__(self, container=scene)


class QKiteToolTransect(QtGui.QDialog):
    def __init__(self, plot):
        QtGui.QDialog.__init__(self)
        log_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/transect.ui')
        loadUi(log_ui, baseinstance=self)

        self.plot = plot
        self.poly_line = None

        self.trans_plot = pg.PlotDataItem(antialias=True,
                                          fillLevel=0.,
                                          fillBrush=pg.mkBrush(0, 127, 0,
                                                               150))

        self.plt_wdgt = pg.PlotWidget()
        self.plt_wdgt.setLabels(bottom={'Distance', 'm'},
                                left={'Displacement', 'm'})
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.enableAutoRange()
        self.plt_wdgt.addItem(self.trans_plot)

        self.layout().addWidget(self.plt_wdgt)
        self.plot.image.sigImageChanged.connect(self.updateTransPlot)
        self.getPolyControl()

    def getPolyControl(self):
        def addPolyLine():
            [[xmin, xmax], [ymin, ymax]] = self.plot.viewRange()
            self.poly_line = pg.PolyLineROI(positions=[(xmin+(xmax-xmin)*.4,
                                                        ymin+(ymax-ymin)*.4),
                                                       (xmin+(xmax-xmin)*.6,
                                                        ymin+(ymax-ymin)*.6)],
                                            pen=pg.mkPen('g', width=2))
            self.plot.addItem(self.poly_line)
            self.updateTransPlot()
            self.poly_line.sigRegionChangeFinished.connect(
                self.updateTransPlot)

        def clearPolyLine():
            try:
                self.plot.removeItem(self.poly_line)
                self.poly_line = None
                self.updateTransPlot()
            except Exception as e:
                print e

        self.createButton.released.connect(addPolyLine)
        self.removeButton.released.connect(clearPolyLine)

    def updateTransPlot(self):
        if self.poly_line is None:
            return

        transect = num.ndarray((0))
        length = 0
        for line in self.poly_line.segments:
            transect = num.append(transect,
                                  line.getArrayRegion(self.plot.image.image,
                                                      self.plot.image))
            p1, p2 = line.listPoints()
            length += (p2-p1).length()
        # interpolate over NaNs
        nans, x = num.isnan(transect), lambda z: z.nonzero()[0]
        transect[nans] = num.interp(x(nans), x(~nans), transect[~nans])
        length = num.linspace(0, length, transect.size)

        self.trans_plot.setData(length, transect)
        self.plt_wdgt.setLimits(xMin=length.min(), xMax=length.max(),
                                yMin=transect.min(), yMax=transect.max()*1.1)
        return


class QKiteParamScene(QKiteParameterGroup):
    def __init__(self, spool, plot, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene'
        self.scene = spool.scene
        self.plot = plot
        self.parameters = {
            'min value': lambda plot: num.nanmin(plot.data),
            'max value': lambda plot: num.nanmax(plot.data),
            'mean value': lambda plot: num.nanmean(plot.data)
        }

        QKiteParameterGroup.__init__(self, self.plot, **kwargs)
        self.plot.image.sigImageChanged.connect(self.updateValues)

        def changeComponent(parameter):
            self.plot.component = parameter.value()

        p = {'name': 'display',
             'values': {
                 'displacement': 'displacement',
                 'theta': 'theta',
                 'phi': 'phi',
                 'thetaDeg': 'thetaDeg',
                 'phiDeg': 'phiDeg',
                 'los.unitE': 'unitE',
                 'los.unitN': 'unitN',
                 'los.unitU': 'unitU',
                 },
             'value': 'displacement'}
        component = pTypes.ListParameter(**p)
        component.sigValueChanged.connect(changeComponent)
        self.pushChild(component, autoIncrementName=None)


class QKiteParamSceneFrame(QKiteParameterGroup):
    def __init__(self, spool, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = '.frame'
        self.frame = spool.scene.frame

        self.parameters = OrderedDict([
            ('cols', None),
            ('rows', None),
            ('llLat', None),
            ('llLon', None),
            ('dLat', None),
            ('dLon', None),
            ('extentE', None),
            ('extentN', None),
            ('spherical_distortion', None),
            ('dN', None),
            ('dE', None),
            ('llNutm', None),
            ('llEutm', None),
            ('utm_zone', None),
            ('utm_zone_letter', None)])

        QKiteParameterGroup.__init__(self, self.frame, **kwargs)


class QKiteParamSceneMeta(QKiteParameterGroup):
    def __init__(self, spool, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = '.meta'
        scene = spool.scene
        self.meta = scene.meta
        self.parameters =\
            [param[0] for param in
             scene.meta.T.inamevals_to_save(scene.meta)]

        QKiteParameterGroup.__init__(self, self.meta, **kwargs)
