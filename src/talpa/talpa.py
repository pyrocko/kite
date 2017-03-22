import sys
from os import path
from PySide import QtCore, QtGui

from .common import ModelDockarea

from ..qt_utils import loadUi
from ..model_scene import ModelScene
from ..models import OkadaSource


model = ModelScene()
src = OkadaSource(
    northing=10000.,
    easting=10000.,
    depth=.0,
    length=5000.,
    width=3000.,
    strike=14.)

model.addSource(src)


def get_resource(filename):
    return path.join(path.dirname(path.realpath(__file__)), 'res', filename)


class Talpa(QtGui.QApplication):
    def __init__(self, *args, **kwargs):
        QtGui.QApplication.__init__(self, ['Talpa'])
        splash_img = QtGui.QPixmap(
            get_resource('talpa_splash.png'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        self.splash = QtGui.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint)

        self.talpa_win = TalpaMainWindow()

        self.updateSplashMessage('')
        self.splash.show()
        self.processEvents()

        self.splash.finish(self.talpa_win)
        self.talpa_win.show()
        rc = self.exec_()
        sys.exit(rc)

    @QtCore.Slot(str)
    def updateSplashMessage(self, msg=''):
        self.splash.showMessage("Loading %s ..." % msg.title(),
                                QtCore.Qt.AlignBottom)


class TalpaMainWindow(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.loadUi()

        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))

        self.openModel(model)

    def loadUi(self):
        loadUi(get_resource('talpa.ui'), baseinstance=self)

    def openModel(self, model):
        m = ModelDockarea(model)
        self.centralwidget.layout().addWidget(m)

    def closeModel(self, model):
        pass
