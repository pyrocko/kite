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
        self.title = 'Scene'
        scene_proxy = spool.scene_proxy

        scene_plot = QKiteScenePlot(scene_proxy)
        self.main_widget = scene_plot
        self.tools = {
            # 'Components': QKiteToolComponents(self.main_widget),
            # 'Displacement Transect': QKiteToolTransect(self.main_widget),
        }

        self.param_scene = QKiteParamScene(scene_proxy, scene_plot)
        self.param_frame = QKiteParamSceneFrame(scene_proxy, expanded=False)
        self.param_meta = QKiteParamSceneMeta(scene_proxy, expanded=False)

        self.param_scene.addChild(self.param_frame)
        self.param_scene.addChild(self.param_meta)

        self.parameters = [self.param_scene]

        self.dialogTransect = QKiteToolTransect(scene_plot, spool)
        spool.actionTransect.triggered.connect(self.dialogTransect.show)
        spool.actionTransect.setEnabled(True)

        scene_proxy.sigSceneModelChanged.connect(self.modelChanged)

        QKiteView.__init__(self)

    def modelChanged(self):
        self.main_widget.update()
        self.main_widget.transFromFrame()

        self.param_scene.updateValues()
        self.param_frame.updateValues()
        self.param_meta.updateValues()

        self.dialogTransect.close()


class QKiteScenePlot(QKitePlot):
    def __init__(self, scene_proxy):
        self.components_available = {
            'displacement':
                ['Scene.displacement', lambda sp: sp.scene.displacement],
            'theta':
                ['Scene.theta', lambda sp: sp.scene.theta],
            'phi':
                ['Scene.phi', lambda sp: sp.scene.phi],
            'thetaDeg':
                ['Scene.thetaDeg', lambda sp: sp.scene.thetaDeg],
            'phiDeg':
                ['Scene.phiDeg', lambda sp: sp.scene.phiDeg],
            'unitE':
                ['Scene.los.unitE', lambda sp: sp.scene.los.unitE],
            'unitN':
                ['Scene.los.unitN', lambda sp: sp.scene.los.unitN],
            'unitU':
                ['Scene.los.unitU', lambda sp: sp.scene.los.unitU],
        }
        self._component = 'displacement'

        QKitePlot.__init__(self, scene_proxy=scene_proxy)
        scene_proxy.sigFrameChanged.connect(self.onFrameChange)
        scene_proxy.sigSceneModelChanged.connect(self.update)

    def onFrameChange(self):
        self.update()
        self.transFromFrame()


class QKiteToolTransect(QtGui.QDialog):
    def __init__(self, plot, parent=None):
        QtGui.QDialog.__init__(self, parent)
        trans_ui = path.join(path.dirname(path.realpath(__file__)),
                             'ui/transect.ui')
        loadUi(trans_ui, baseinstance=self)
        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))
        self.createButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_ArrowUp))
        self.removeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogDiscardButton))

        self.plot = plot
        self.poly_line = None

        self.trans_plot = pg.PlotDataItem(antialias=True,
                                          fillLevel=0.,
                                          fillBrush=pg.mkBrush(0, 127, 0,
                                                               150))

        self.plt_wdgt = pg.PlotWidget()
        self.plt_wdgt.setLabels(bottom={'Distance', 'm'},
                                left='Displacement [m]')
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.enableAutoRange()
        self.plt_wdgt.addItem(self.trans_plot)

        self.layout().addWidget(self.plt_wdgt)
        self.plot.image.sigImageChanged.connect(self.updateTransPlot)
        self.createButton.released.connect(self.addPolyLine)
        self.removeButton.released.connect(self.removePolyLine)

        parent.scene_proxy.sigConfigChanged.connect(self.close)

    def addPolyLine(self):
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

    def removePolyLine(self):
        if self.poly_line is None:
            return

        self.plot.removeItem(self.poly_line)
        self.poly_line = None
        self.updateTransPlot()

    def closeEvent(self, event):
        self.removePolyLine()

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
    def __init__(self, scene_proxy, plot, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene'
        self.plot = plot

        self.parameters = {
            'min value': lambda plot: num.nanmin(plot.data),
            'max value': lambda plot: num.nanmax(plot.data),
            'mean value': lambda plot: num.nanmean(plot.data),
        }

        self.plot.image.sigImageChanged.connect(self.updateValues)

        QKiteParameterGroup.__init__(self,
                                     model=self.plot,
                                     model_attr=None,
                                     **kwargs)

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
        self.pushChild(component)


class QKiteParamSceneFrame(QKiteParameterGroup):
    def __init__(self, scene_proxy, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = '.frame'

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
            ('utm_zone_letter', None),
            ])

        scene_proxy.sigFrameChanged.connect(self.updateValues)

        QKiteParameterGroup.__init__(self,
                                     model=scene_proxy,
                                     model_attr='frame',
                                     **kwargs)


class QKiteParamSceneMeta(QKiteParameterGroup):
    def __init__(self, scene_proxy, **kwargs):
        from datetime import datetime as dt
        kwargs['type'] = 'group'
        kwargs['name'] = '.meta'

        def str_to_time(d, fmt='%Y-%m-%d %H:%M:%S'):
            return dt.strftime(dt.fromtimestamp(d), fmt)

        self.parameters = OrderedDict([
            ('scene_title',
             lambda sc: sc.meta.scene_title),
            ('scene_id',
             lambda sc: sc.meta.scene_id),
            ('satellite_name',
             lambda sc: sc.meta.satellite_name),
            ('orbit_direction',
             lambda sc: sc.meta.orbit_direction),
            ('time_master',
             lambda sc: str_to_time(sc.meta.time_master)),
            ('time_slave',
             lambda sc: str_to_time(sc.meta.time_slave)),
            ('time_separation',
             lambda sc: str_to_time(sc.meta.time_separation,
                                    '%j days %H:%m hours')),
            ])

        scene_proxy.sigConfigChanged.connect(self.updateValues)

        QKiteParameterGroup.__init__(self,
                                     model=scene_proxy,
                                     model_attr='scene',
                                     **kwargs)
