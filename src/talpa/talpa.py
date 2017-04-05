import sys
from PySide import QtCore, QtGui

from .common import get_resource
from .multiplot import PlotDockarea
from .sources_dock import SourcesListDock

from sandbox_model import SandboxModel
from ..model_scene import ModelScene
from ..qt_utils import loadUi, SceneLog, validateFilename


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
        loadUi(get_resource('talpa.ui'), baseinstance=self)

        self.sandbox = SandboxModel.empty()

        self.log = SceneLog(self, self.sandbox)

        self.actionSaveModel.triggered.connect(
            self.onSaveModel)
        self.actionLoadModel.triggered.connect(
            self.onLoadModel)
        self.actionExportKiteScene.triggered.connect(
            self.onExportScene)

        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))
        self.actionAbout_Talpa.triggered.connect(
            self.aboutDialog().show)

        self.actionLog.triggered.connect(
            self.log.show)

        self.createView(self.sandbox)

    def aboutDialog(self):
        self._about = QtGui.QDialog()
        loadUi(get_resource('about.ui'), baseinstance=self._about)
        return self._about

    def createView(self, sandbox):
        plots = PlotDockarea(sandbox)
        sources = SourcesListDock(sandbox, parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sources)
        self.centralwidget.layout().addWidget(plots)

    def onSaveModel(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML *.yml (*.yml)',
            caption='Save ModelScene')
        if not validateFilename(filename):
            return
        self.sandbox.model.save(filename)

    def onLoadModel(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML *.yml (*.yml)',
            caption='Load ModelScene')
        if not validateFilename(filename):
            return
        model = ModelScene.load(filename)
        self.sandbox.setModel(model)

    def onExportScene(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Save scene')
        if not validateFilename(filename):
            return
        scene = self.sandbox.model.getKiteScene()
        scene.save(filename)

    def closeModel(self, sandbox):
        pass
