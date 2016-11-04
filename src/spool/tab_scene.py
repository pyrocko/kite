#!/usr/bin/python2
from PySide import QtGui
import pyqtgraph as pg
import numpy as num
from .tab import QKiteDock, QKiteToolComponents, QKitePlot

__all__ = ['QKiteSceneDock']


class QKiteSceneDock(QKiteDock):
    def __init__(self, scene):
        self.title = 'Displacement Scene'
        self.main_widget = QKiteScenePlot
        self.tools = {
            'Components': QKiteToolComponents,
            # 'Colormap': QKiteToolGradient,
            'Transect': QKiteToolTransect,
            # 'Histogram': QKiteToolHistogram,
        }

        QKiteDock.__init__(self, scene)


class QKiteScenePlot(QKitePlot):
    def __init__(self, scene):

        self.components_available = {
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

        QKitePlot.__init__(self, container=scene)


class QKiteToolTransect(QtGui.QWidget):
    def __init__(self, plot):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        self.trans_plot = pg.PlotDataItem(antialias=True,
                                          fillLevel=0.,
                                          fillBrush=pg.mkBrush(0, 127, 0,
                                                               150))

        self.plt_wdgt = pg.PlotWidget(labels={'bottom': 'Distance / m',
                                              'left': 'Displacement / m'})
        self.plt_wdgt.showGrid(True, True, alpha=.5)
        self.plt_wdgt.addItem(self.trans_plot)
        self.plt_wdgt.enableAutoRange()

        self.poly_line = None

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
                                  line.getArrayRegion(self.plot.data,
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
