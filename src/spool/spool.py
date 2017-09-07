#!/usr/bin/python
from PySide import QtGui, QtCore
import sys
import time  # noqa
import pyqtgraph as pg

from kite.qt_utils import loadUi, validateFilename, SceneLog
from kite.scene import Scene

from .scene_model import SceneModel
from .tab_scene import KiteScene
from .tab_quadtree import KiteQuadtree  # noqa
from .tab_covariance import KiteCovariance  # noqa
from .base import get_resource


class Spool(QtGui.QApplication):
    def __init__(self, scene=None, import_data=None, load_file=None):
        QtGui.QApplication.__init__(self, ['Spool'])
        # self.setStyle('plastique')
        splash_img = QtGui.QPixmap(get_resource('spool_splash.png'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        self.splash = QtGui.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint)
        self.updateSplashMessage('Scene')
        self.splash.show()
        self.processEvents()

        self.spool_win = SpoolMainWindow()
        self.spool_win.sigLoadingModule.connect(self.updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)
        self.aboutToQuit.connect(self.spool_win.model.worker_thread.quit)
        self.aboutToQuit.connect(self.spool_win.model.deleteLater)
        self.aboutToQuit.connect(self.splash.deleteLater)
        self.aboutToQuit.connect(self.deleteLater)

        if scene is not None:
            self.addScene(scene)
        if import_data is not None:
            self.importScene(import_data)
        if load_file is not None:
            self.loadScene(load_file)

        self.splash.finish(self.spool_win)
        self.spool_win.show()
        rc = self.exec_()
        sys.exit(rc)

    @QtCore.Slot(str)
    def updateSplashMessage(self, msg=''):
        self.splash.showMessage("Loading %s ..." % msg.title(),
                                QtCore.Qt.AlignBottom)

    def addScene(self, scene):
        self.spool_win.addScene(scene)

    def importScene(self, filename):
        self.spool_win.model.importFile(filename)

    def loadScene(self, filename):
        self.spool_win.model.loadFile(filename)

    def __del__(self):
        pass


class SpoolMainWindow(QtGui.QMainWindow):
    sigImportFile = QtCore.Signal(str)
    sigLoadFile = QtCore.Signal(str)
    sigLoadConfig = QtCore.Signal(str)
    sigExportWeightMatrix = QtCore.Signal(str)
    sigLoadingModule = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.loadUi()

        self.views = [KiteScene, KiteQuadtree, KiteCovariance]

        self.ptree = KiteParameterTree(showHeader=False)
        self.ptree_dock = QtGui.QDockWidget('Parameters', self)
        self.ptree_dock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                                    QtGui.QDockWidget.DockWidgetMovable)
        self.ptree_dock.setWidget(self.ptree)
        self.addDockWidget(
            QtCore.Qt.LeftDockWidgetArea, self.ptree_dock)

        self.model = SceneModel()
        self.model.sigSceneModelChanged.connect(
            self.buildViews)

        self.sigLoadFile.connect(
            self.model.loadFile)
        self.sigImportFile.connect(
            self.model.importFile)
        self.sigLoadConfig.connect(
            self.model.loadConfig)
        self.sigExportWeightMatrix.connect(
            self.model.exportWeightMatrix)

        self.actionSave_config.triggered.connect(
            self.onSaveConfig)
        self.actionSave_scene.triggered.connect(
            self.onSaveScene)
        self.actionLoad_config.triggered.connect(
            self.onLoadConfig)
        self.actionLoad_scene.triggered.connect(
            self.onOpenScene)

        self.actionImport_scene.triggered.connect(
            self.onImportScene)
        self.actionExport_quadtree.triggered.connect(
            self.onExportQuadtree)
        self.actionExport_weights.triggered.connect(
            self.onExportWeightMatrix)

        self.actionAbout_Spool.triggered.connect(
            self.aboutDialog().show)
        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))

        self.log = SceneLog(self, self.model)
        self.actionLog.triggered.connect(
            self.log.show)

        self.progress = QtGui.QProgressDialog('', None, 0, 0, self)
        self.progress.setValue(0)
        self.progress.closeEvent = lambda ev: ev.ignore()
        self.progress.setMinimumWidth(400)
        self.progress.setWindowTitle('processing...')
        self.model.sigProcessingFinished.connect(self.progress.reset)

    def aboutDialog(self):
        self._about = QtGui.QDialog()
        loadUi(get_resource('about.ui'), baseinstance=self._about)
        return self._about

    def loadUi(self):
        loadUi(get_resource('spool.ui'), baseinstance=self)

    def addScene(self, scene):
        self.model.setScene(scene)
        self.buildViews()

    def buildViews(self):
        title = self.model.scene.meta.filename or 'Untitled'
        self.setWindowTitle('Spool - %s' % title)
        if self.model.scene is None or self.tabs.count() != 0:
            return
        for v in self.views:
            self.addView(v)
        self.model.sigProcessingStarted.connect(self.processingStarted)

    def addView(self, view):
        self.sigLoadingModule.emit(view.title)
        view = view(self)
        QtCore.QCoreApplication.processEvents()
        self.tabs.addTab(view, view.title)
        QtCore.QCoreApplication.processEvents()
        if hasattr(view, 'parameters'):
            for parameter in view.parameters:
                self.ptree.addParameters(parameter)

    @QtCore.Slot(str)
    def processingStarted(self, text):
        self.progress.setLabelText(text)
        self.progress.show()

    def onSaveConfig(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML file *.yml (*.yml)', caption='Save scene YAML config')
        if not validateFilename(filename):
            return
        self.model.scene.saveConfig(filename)

    def onSaveScene(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Save scene')
        if not validateFilename(filename):
            return
        self.model.scene.save(filename)

    def onLoadConfig(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML file *.yml (*.yml)', caption='Load scene YAML config')
        if not validateFilename(filename):
            return
        self.sigLoadConfig.emit(filename)

    def onOpenScene(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Load kite scene')
        if not validateFilename(filename):
            return
        self.model.loadFile(filename)

    def onImportScene(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            self,
            filter='Supported Formats, *.grd, *.geo, *unw*, *.mat '
                   '(*.grd,*.geo,*unw*,*.mat);;'
                   'GMT5SAR Scene, *.grd (*.grd);;'
                   'ISCE Scene, *unw* (*unw*);;Gamma Scene *.geo (*.geo);;'
                   'Matlab Container, *.mat (*.mat);;Any File * (*)',
            caption='Import scene to spool')
        if not validateFilename(filename):
            return
        self.sigImportFile.emit(filename)

    def onExportQuadtree(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='CSV File *.csv (*.csv)', caption='Export Quadtree CSV')
        if not validateFilename(filename):
            return
        self.model.quadtree.export(filename)

    def onExportWeightMatrix(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='Text File *.txt (*.txt)',
            caption='Export Covariance Weights',)
        if not validateFilename(filename):
            return
        self.sigExportWeightMatrix.emit(filename)

    def exit(self):
        pass


class KiteParameterTree(pg.parametertree.ParameterTree):
    pass


__all__ = ['Spool']

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    Spool(scene=sc)
