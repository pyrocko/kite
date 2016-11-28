#!/usr/bin/python2
from PySide import QtGui
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
import numpy as num
from .tab import QKiteDock, QKiteToolComponents, QKitePlot  # noqa

__all__ = ['QKiteSceneDock']


class QKiteSceneDock(QKiteDock):
    def __init__(self, scene):
        self.title = 'Scene.displacement'
        self.main_widget = QKiteScenePlot(scene)
        self.tools = {
            # 'Components': QKiteToolComponents(self.main_widget),
            'Displacement Transect': QKiteToolTransect(self.main_widget),
        }

        self.parameters = [QKiteSceneParam(self.main_widget),
                           QKiteSceneParamFrame(scene),
                           QKiteSceneParamMeta(scene)]

        QKiteDock.__init__(self, scene)


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


class QKiteToolTransect(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
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

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.plt_wdgt)
        self.layout.addWidget(self.getPolyControl())

        self.plot.image.sigImageChanged.connect(self.updateTransPlot)
        # self.plot.image.sigImageChanged.connect(self.addPolyLine)

    def getPolyControl(self):
        wdgt = QtGui.QWidget()

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

        btn_addPoly = QtGui.QPushButton('Create Transsect')
        btn_addPoly.clicked.connect(addPolyLine)
        btn_clearPoly = QtGui.QPushButton('Clear Transsect')
        btn_clearPoly.clicked.connect(clearPolyLine)

        layout = QtGui.QHBoxLayout(wdgt)
        layout.addWidget(btn_addPoly)
        layout.addWidget(btn_clearPoly)

        return wdgt

    def updateTransPlot(self):
        if self.poly_line is None:
            self.trans_plot.setData((0))
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


class QKiteSceneParam(pTypes.GroupParameter):
    def __init__(self, main_widget, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene'
        self.plot = main_widget
        pTypes.GroupParameter.__init__(self, **kwargs)

        opts = {'name': 'component',
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
                'value': 'displacement',
                }
        component = pTypes.ListParameter(**opts)

        def changeComponent(parameter):
            self.plot.component = parameter.value()

        component.sigValueChanged.connect(changeComponent)
        self.addChild(component, autoIncrementName=None)


class QKiteSceneParamFrame(pTypes.GroupParameter):
    def __init__(self, scene, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.frame'
        self.scene = scene
        pTypes.GroupParameter.__init__(self, **kwargs)

        for param in self.scene.frame._parameters:
            value = getattr(self.scene.frame, param)
            if isinstance(value, float):
                value = '%.4f' % value
            else:
                value = str(value)
            self.addChild({'name': param,
                           'value': value,
                           'type': 'str',
                           'suffix': 'm',
                           'readonly': True})


class QKiteSceneParamMeta(pTypes.GroupParameter):
    def __init__(self, scene, **kwargs):
        kwargs['type'] = 'group'
        kwargs['name'] = 'Scene.meta'
        self.scene = scene
        pTypes.GroupParameter.__init__(self, **kwargs)

        for param, value in self.scene.meta.T.inamevals_to_save(
         self.scene.meta):
            self.addChild({'name': param,
                           'value': value,
                           'type': 'str',
                           'readonly': True})


class QKiteSceneParameters(pTypes.GroupParameter):
    def __init__(self, **kwargs):
        pass
