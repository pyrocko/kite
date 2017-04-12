import sys
from PySide import QtCore, QtGui

from .common import get_resource
from .multiplot import ModelSceneDockarea, ModelReferenceDockarea
from .sources_dock import SourcesListDock

from sandbox_model import SandboxModel
from .tool_dialogs import ExtentDialog
from ..model_scene import ModelScene
from ..qt_utils import loadUi, SceneLog, validateFilename


class Talpa(QtGui.QApplication):
    def __init__(self, filename=None):
        QtGui.QApplication.__init__(self, ['Talpa'])
        splash_img = QtGui.QPixmap(
            get_resource('talpa_splash.png'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        self.splash = QtGui.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint)

        self.talpa_win = TalpaMainWindow(filename=filename)

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
        filename = kwargs.pop('filename', None)

        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource('talpa.ui'), baseinstance=self)

        self.sandbox = SandboxModel.empty()

        self.log = SceneLog(self, self.sandbox)

        self.actionSaveModel.triggered.connect(
            self.onSaveModel)
        self.actionLoadModel.triggered.connect(
            self.loadModel)
        self.actionExportKiteScene.triggered.connect(
            self.onExportScene)
        self.actionChangeExtent.triggered.connect(
            self.extentDialog)
        self.actionLoadReferenceScene.triggered.connect(
            self.onLoadReferenceScene)

        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))
        self.actionAbout_Talpa.triggered.connect(
            self.aboutDialog().show)
        self.actionLog.triggered.connect(
            self.log.show)

        self.sandbox.sigModelChanged.connect(
            self.createMisfitWindow)

        self.progress = QtGui.QProgressDialog('', None, 0, 0, self)
        self.progress.setValue(0)
        self.progress.closeEvent = lambda ev: ev.ignore()
        self.progress.setMinimumWidth(400)
        self.progress.setWindowTitle('processing...')
        self.sandbox.sigProcessingFinished.connect(
            self.processingFinished)
        self.sandbox.sigProcessingStarted.connect(
            self.processingStarted)

        if filename is not None:
            self.loadModel(filename)
        self.createView(self.sandbox)

    def createView(self, sandbox):
        plots = ModelSceneDockarea(sandbox)
        sources = SourcesListDock(sandbox, parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sources)
        self.centralwidget.layout().addWidget(plots)

    def aboutDialog(self):
        self._about = QtGui.QDialog()
        loadUi(get_resource('about.ui'), baseinstance=self._about)
        return self._about

    def extentDialog(self):
        ExtentDialog(self.sandbox, self).show()

    @QtCore.Slot(str)
    def processingStarted(self, text):
        self.progress.setLabelText(text)
        self.progress.show()

    @QtCore.Slot()
    def processingFinished(self):
        self.progress.reset()
        self.progress.close()

    def onSaveModel(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML *.yml (*.yml)',
            caption='Save ModelScene')
        if not validateFilename(filename):
            return
        self.sandbox.model.save(filename)

    def loadModel(self, filename=None):
        if filename is None:
            filename, _ = QtGui.QFileDialog.getOpenFileName(
                filter='YAML *.yml (*.yml)',
                caption='Load ModelScene')
        if not validateFilename(filename):
            return
        model = ModelScene.load(filename)
        self.sandbox.setModel(model)

    def onLoadReferenceScene(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML *.yml (*.yml)',
            caption='Load kite.Scene')
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
            lambda: self.actionMisfitScene.setChecked(False))
        self.actionMisfitScene.toggled.connect(toggleWindow)
        self.actionMisfitScene.setEnabled(True)

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


class MisfitWindow(QtGui.QMainWindow):
    windowClosed = QtCore.Signal()

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource('window_reference.ui'), self)

        self.move(
            self.parent().window().mapToGlobal(
                self.parent().window().rect().center()) -
            self.mapToGlobal(self.rect().center()))

        self.sandbox = sandbox

        self.actionOptimizeSource.triggered.connect(
            self.sandbox.optimizeSource)

        self.createView(self.sandbox)

    def createView(self, sandbox):
        plots = ModelReferenceDockarea(sandbox)
        self.centralwidget.layout().addWidget(plots)

    def closeEvent(self, ev):
        self.windowClosed.emit()
        ev.accept()
