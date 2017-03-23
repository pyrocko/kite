import sys
from os import path
from PySide import QtCore, QtGui

from .multiplot import PlotDockarea
from .sources_dock import SourcesListDock

from model_proxy import SandboxModel
from ..qt_utils import loadUi


sandbox = SandboxModel.randomOkada(4)


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

        self.aboutToQuit.connect(self.splash.deleteLater)
        self.aboutToQuit.connect(self.deleteLater)

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

        self.openModel(sandbox)

    def loadUi(self):
        loadUi(get_resource('talpa.ui'), baseinstance=self)

    def openModel(self, sandbox):
        plots = PlotDockarea(sandbox)
        sources = SourcesListDock(sandbox, parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sources)
        self.centralwidget.layout().addWidget(plots)

    def closeModel(self, sandbox):
        pass
