#!/bin/python
from PySide import QtGui
from PySide import QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg \
                                        as FigureCanvas
from matplotlib.figure import Figure


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


class QKiteDisplacementPlot2D(FigureCanvas, _QKiteScene):
    def __init__(self, parent=None, scene=None):

        self.fig = Figure(frameon=False, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        _QKiteScene.__init__(self, scene)
        # self.axes.hold(False)
        self.axes = self.fig.add_subplot(111)

        self.plot = self.scene.plot
        self.plot.plot(axes=self.axes)

        self.plot.subscribe(self.updatePlot)

    def updatePlot(self):
        self.fig.canvas.draw()


class QKiteDisplacementControl(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        QKiteColormapControl(self, self.scene.plot)


class _QKiteSlider(QtGui.QSlider, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QSlider.__init__(self, parent)
        _QKiteScene.__init__(self, scene)


class QKiteColormapControl(QtGui.QHBoxLayout):
    def __init__(self, parent=None, plot=None):
        QtGui.QHBoxLayout.__init__(self, parent)
        if not parent.__dict__.get('plot', None) and plot is None:
            raise AttributeError('Plot2D instance must be defined')
        self.plot = plot or parent.plot

        label = QtGui.QLabel('Colormap')

        self.addWidget(label)
        self.addComboBox()

    def addComboBox(self):
        self.cm_cbox = QtGui.QComboBox()
        for cmtype, cmlist in self.plot._availableColormaps():
            self.cm_cbox.addItem(cmtype)
            _i = self.cm_cbox.findText(cmtype)
            self.cm_cbox.setItemData(_i, '', role=QtCore.Qt.UserRole-1)

            self.cm_cbox.addItems(cmlist)

        _i = self.cm_cbox.findText(self.plot.image.get_cmap().name)
        self.cm_cbox.setCurrentIndex(_i)

        def change_colormap(index):
            self.plot.setColormap(self.cm_cbox.itemText(index))

        self.cm_cbox.currentIndexChanged.connect(change_colormap)
        self.addWidget(self.cm_cbox)

    def addSlider(self):
        pass


class QKiteColormapComboBox(QtGui.QComboBox, _QKiteScene):
    pass


class QKiteDisplacement(QtGui.QWidget, _QKiteScene):
    def __init__(self, parent=None, scene=None):
        QtGui.QWidget.__init__(self, parent)
        _QKiteScene.__init__(self, scene)

        QKiteDisplacementPlot2D(self)
        QKiteDisplacementControl(self)


if __name__ == '__main__':
    import sys
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()

    qt_app = QtGui.QApplication(sys.argv)
    app = QKiteDisplacement(scene=sc)
    app.show()
    qt_app.exec_()
