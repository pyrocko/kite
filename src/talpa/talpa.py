import sys
from PySide import QtCore, QtGui

from .common import get_resource
from .multiplot import PlotDockarea
from .sources_dock import SourcesListDock

from sandbox_model import SandboxModel
from ..qt_utils import loadUi, SceneLog


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

        self.sandbox = SandboxModel.empty()

        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))

        self.log = SceneLog(self, self.sandbox)
        self.actionLog.triggered.connect(
            self.log.show)

        self.openModel(self.sandbox)

    def loadUi(self):
        loadUi(get_resource('talpa.ui'), baseinstance=self)

    def openModel(self, sandbox):
        plots = PlotDockarea(sandbox)
        sources = SourcesListDock(sandbox, parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sources)
        self.centralwidget.layout().addWidget(plots)

    def closeModel(self, sandbox):
        pass
