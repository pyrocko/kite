#!/bin/python
from PySide import QtGui
from PySide import QtCore
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg \
                                        as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['backend.qt4'] = 'PySide'


class _QKiteScene(object):
    def __init__(self, scene=None):
        if scene is None:
            try:
                if self.parent().__dict__.get('scene', None) is not None:
                    self.scene = self.parent().__dict__.get('scene')
                    return
            except Exception as e:
                print e
        else:
            self.scene = scene
            return
        raise AttributeError('Scene must be defined')


class QKiteScenePlot2D(FigureCanvas, _QKiteScene):
    def __init__(self, parent=None, scene=None):

        self.fig = Figure(frameon=False, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        _QKiteScene.__init__(self, scene)
        # self.axes.hold(False)
        self.axes = self.fig.add_subplot(111)

        self.plot = self.scene.plot
        self.plot.plot(axes=self.axes)

        def updatePlot():
            self.fig.canvas.draw()

        self.plot.subscribe(updatePlot)


class QKiteControlScene(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        QKiteControlColormap(self, self.scene.plot)


class QKiteControlColormap(QtGui.QHBoxLayout):
    def __init__(self, parent=None, plot=None):
        QtGui.QHBoxLayout.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        # label = QtGui.QLabel('Colormap')

        # self.addWidget(label)
        self.addColormapsComboBox()

    def addColormapsComboBox(self):
        self.cm_combo = QtGui.QComboBox()
        for cmtype, cmlist in self.plot._availableColormaps():
            self.cm_combo.addItem('%s' % cmtype)
            _i = self.cm_combo.findText('%s' % cmtype)
            self.cm_combo.setItemData(_i, '', role=QtCore.Qt.UserRole-1)
            for cm in cmlist:
                self.cm_combo.addItem(' %s' % cm, cm)

        _i = self.cm_combo.findData(self.plot.image.get_cmap().name)
        self.cm_combo.setCurrentIndex(_i)

        def changeColormap(index):
            self.plot.setColormap(self.cm_cbox.itemText(index))

        self.cm_cbox.currentIndexChanged.connect(changeColormap)
        self.addWidget(self.cm_cbox)

    def addSlider(self):
        self.cm_slider = QtGui.QSlider()
        self.addWidget(self.cm_combo)

    def addColormapRangeSlider(self):
        self.cm_slider = QtGui.QSlider()
        self.addWidget(self.cm_slider)

    def addColormapAutoChecker(self):
        self.cm_auto = QtGui.QCheckBox()
        self.cm_auto.setText('Auto range')
        self.addWidget(self.cm_auto)

    def addColormapSymmetricChecker(self):
        self.cm_sym = QtGui.QCheckBox()
        self.cm_sym.setText('Symetric range')
        self.addWidget(self.cm_sym)


class QKiteControlComponent(QtGui.QWidget):
    pass


class QKiteScene(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        QKiteScenePlot2D(self)
        tab1 = QtGui.QDockWidget('Colorbar', self)
        tab1.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        tab1.setWidget(QKiteControlScene(parent=tab1, scene=scene))
        tab1.setFloating(True)
        tab1.setFeatures(QtGui.QDockWidget.DockWidgetFeature.DockWidgetMovable)

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()

    qt_app = QtGui.QApplication(['KiteSpool'])
    app = QKiteScene(scene=sc)
    app.show()
    qt_app.exec_()
