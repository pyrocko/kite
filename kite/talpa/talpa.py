import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from kite.qt_utils import SceneLog, loadUi, validateFilename
from kite.sandbox_scene import SandboxScene

from .config import ConfigDialog
from .multiplot import ModelReferenceDockarea, SandboxSceneDockarea
from .sandbox_model import SandboxModel
from .sources_dock import SourcesListDock
from .tool_dialogs import ExtentDialog, LosDialog
from .util import get_resource


class Talpa(QtWidgets.QApplication):
    def __init__(self, filename=None):
        QtWidgets.QApplication.__init__(self, ["Talpa"])

        splash_img = QtGui.QPixmap(get_resource("talpa_splash.png")).scaled(
            QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio
        )
        self.splash = QtWidgets.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint
        )
        self.updateSplashMessage("Talpa")
        self.splash.show()
        self.processEvents()

        self.talpa_win = TalpaMainWindow(filename=filename)

        self.talpa_win.actionExit.triggered.connect(self.exit)
        self.aboutToQuit.connect(self.talpa_win.sandbox.worker_thread.quit)
        self.aboutToQuit.connect(self.talpa_win.sandbox.deleteLater)
        self.aboutToQuit.connect(self.splash.deleteLater)
        self.aboutToQuit.connect(self.deleteLater)

        self.talpa_win.show()

        self.splash.finish(self.talpa_win)
        rc = self.exec_()
        sys.exit(rc)

    @QtCore.pyqtSlot(str)
    def updateSplashMessage(self, msg=""):
        self.splash.showMessage("Loading %s ..." % msg.title(), QtCore.Qt.AlignBottom)


class TalpaMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        filename = kwargs.pop("filename", None)
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource("talpa.ui"), baseinstance=self)
        self.sandbox = SandboxModel.empty()

        self.log = SceneLog(self, self.sandbox)

        self.actionSaveModel.triggered.connect(self.onSaveModel)
        self.actionLoadModel.triggered.connect(self.loadModel)
        self.actionExportKiteScene.triggered.connect(self.onExportScene)
        self.actionChangeExtent.triggered.connect(self.extentDialog)
        self.actionChangeLos.triggered.connect(self.losDialog)
        self.actionLoadReferenceScene.triggered.connect(self.onLoadReferenceScene)

        self.actionConfiguration.triggered.connect(self.configDialog)

        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl("https://pyrocko.org")
        )
        self.actionAbout_Talpa.triggered.connect(self.aboutDialog().show)
        self.actionLog.triggered.connect(self.log.show)

        self.sandbox.sigModelChanged.connect(self.createMisfitWindow)

        if filename is not None:
            self.loadModel(filename)
        self.createView(self.sandbox)

    def createView(self, sandbox):
        plots = SandboxSceneDockarea(sandbox)
        sources = SourcesListDock(sandbox, parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sources)
        self.centralwidget.layout().addWidget(plots)

    def aboutDialog(self):
        self._about = QtWidgets.QDialog(self)
        loadUi(get_resource("about.ui"), baseinstance=self._about)
        return self._about

    @QtCore.pyqtSlot()
    def extentDialog(self):
        ExtentDialog(self.sandbox, self).show()

    @QtCore.pyqtSlot()
    def losDialog(self):
        LosDialog(self.sandbox, self).show()

    @QtCore.pyqtSlot()
    def configDialog(self):
        ConfigDialog(self).show()

    @QtCore.pyqtSlot(str)
    def processingStarted(self, text):
        self.progress.setLabelText(text)
        self.progress.show()

    @QtCore.pyqtSlot()
    def onSaveModel(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter="YAML *.yml (*.yml)", caption="Save SandboxScene"
        )
        if not validateFilename(filename):
            return
        self.sandbox.model.save(filename)

    @QtCore.pyqtSlot()
    def loadModel(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                filter="YAML *.yml (*.yml)", caption="Load SandboxScene"
            )
        if not validateFilename(filename):
            return
        model = SandboxScene.load(filename)
        self.sandbox.setModel(model)

    def onLoadReferenceScene(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter="YAML *.yml (*.yml)", caption="Load kite.Scene"
        )
        if not validateFilename(filename):
            return
        self.sandbox.model.loadReferenceScene(filename)

        self.createMisfitWindow()
        self.actionMisfitScene.setChecked(True)

    def createMisfitWindow(self):
        if self.sandbox.model.reference is None:
            return

        self.misfitWindow = MisfitWindow(self.sandbox, self)

        def toggleWindow(switch):
            if switch:
                self.misfitWindow.show()
            else:
                self.misfitWindow.close()

        self.misfitWindow.windowClosed.connect(
            lambda: self.actionMisfitScene.setChecked(False)
        )
        self.actionMisfitScene.toggled.connect(toggleWindow)
        self.actionMisfitScene.setEnabled(True)

    def onExportScene(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter="YAML *.yml and NumPy container *.npz (*.yml *.npz)",
            caption="Save scene",
        )
        if not validateFilename(filename):
            return
        scene = self.sandbox.model.getKiteScene()
        scene.save(filename)

    def closeModel(self, sandbox):
        pass


class MisfitWindow(QtWidgets.QMainWindow):
    windowClosed = QtCore.pyqtSignal()

    def __init__(self, sandbox, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource("window_reference.ui"), self)

        self.move(
            self.parent().window().mapToGlobal(self.parent().window().rect().center())
            - self.mapToGlobal(self.rect().center())
        )

        self.sandbox = sandbox

        self.actionOptimizeSource.triggered.connect(self.sandbox.optimizeSource)

        self.createView(self.sandbox)

    def createView(self, sandbox):
        plots = ModelReferenceDockarea(sandbox)
        self.centralwidget.layout().addWidget(plots)

    def closeEvent(self, ev):
        self.windowClosed.emit()
        ev.accept()
