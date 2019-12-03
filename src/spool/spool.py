#!/usr/bin/python
from PyQt5 import QtGui, QtCore, QtWidgets
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


class Spool(QtWidgets.QApplication):
    def __init__(self, scene=None, import_file=None, load_file=None):
        QtWidgets.QApplication.__init__(self, ['Spool'])
        # self.setStyle('plastique')
        splash_img = QtGui.QPixmap(get_resource('spool_splash.png'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        self.splash = QtWidgets.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint)
        self.updateSplashMessage('Scene')
        self.splash.show()
        self.processEvents()

        self.spool_win = SpoolMainWindow()
        self.spool_win.sigLoadingModule.connect(self.updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)
        self.aboutToQuit.connect(self.spool_win.model.worker_thread.quit,
                                 type=QtCore.Qt.QueuedConnection)
        self.aboutToQuit.connect(self.spool_win.model.deleteLater)
        self.aboutToQuit.connect(self.splash.deleteLater)
        self.aboutToQuit.connect(self.deleteLater)

        if scene is not None:
            self.addScene(scene)
        elif import_file is not None:
            self.importScene(import_file)
        elif load_file is not None:
            self.loadScene(load_file)

        self.splash.finish(self.spool_win)
        self.spool_win.show()

    @QtCore.pyqtSlot(str)
    def updateSplashMessage(self, msg=''):
        self.splash.showMessage(
            'Loading %s ...' % msg.title(), QtCore.Qt.AlignBottom)

    def addScene(self, scene):
        self.spool_win.addScene(scene)

    def importScene(self, filename):
        self.spool_win.model.importFile(filename)

    def loadScene(self, filename):
        self.spool_win.model.loadFile(filename)


class SpoolMainWindow(QtWidgets.QMainWindow):
    VIEWS = [KiteScene, KiteQuadtree, KiteCovariance]

    sigImportFile = QtCore.pyqtSignal(str)
    sigLoadFile = QtCore.pyqtSignal(str)
    sigLoadConfig = QtCore.pyqtSignal(str)
    sigExportWeightMatrix = QtCore.pyqtSignal(str)
    sigLoadingModule = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource('spool.ui'), baseinstance=self)

        self.views = []
        self.active_view = None

        self.ptree = KiteParameterTree(showHeader=False)
        self.ptree_dock = QtWidgets.QDockWidget('Parameters', self)
        self.ptree_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetMovable)
        self.ptree_dock.setWidget(self.ptree)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.ptree_dock)

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
        self.actionAbout_Qt5.triggered.connect(
            lambda: QtGui.QMessageBox.aboutQt(self))
        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('https://pyrocko.org'))

        self.log = SceneLog(self, self.model)
        self.actionLog.triggered.connect(
            self.log.show)

        self.progress = QtWidgets.QProgressDialog('', None, 0, 0, self)
        self.progress.setValue(0)
        self.progress.closeEvent = lambda ev: ev.ignore()
        self.progress.setMinimumWidth(400)
        self.progress.setWindowTitle('Processing...')
        self.progress.reset()
        self.progress_timer = None

    def aboutDialog(self):
        self._about = QtGui.QDialog(self)
        loadUi(get_resource('about.ui'), baseinstance=self._about)
        return self._about

    def addScene(self, scene):
        self.model.setScene(scene)
        self.buildViews()

    def buildViews(self):
        title = self.model.scene.meta.filename or 'Untitled'
        self.setWindowTitle('Spool - %s' % title)
        if self.model.scene is None or self.tabs.count() != 0:
            return
        for v in self.VIEWS:
            self.addView(v)
        self.model.sigProcessingStarted.connect(
            self.processingStarted,
            type=QtCore.Qt.QueuedConnection)
        self.model.sigProcessingFinished.connect(
            self.processingFinished,
            type=QtCore.Qt.QueuedConnection)

        self.tabs.currentChanged.connect(self.activateView)

        self.activateView(0)

    def addView(self, view):
        self.sigLoadingModule.emit(view.title)
        view = view(self)

        QtCore.QCoreApplication.processEvents()
        self.tabs.addTab(view, view.title)
        QtCore.QCoreApplication.processEvents()

        if hasattr(view, 'parameters'):
            for parameter in view.parameters:
                self.ptree.addParameters(parameter)

        self.views.append(view)

    @QtCore.pyqtSlot(int)
    def activateView(self, index):
        if self.active_view is not None:
            self.active_view.deactivateView()

        self.active_view = self.views[index]
        self.active_view.activateView()

    @QtCore.pyqtSlot(str)
    def processingStarted(self, text):
        quadtree = self.model.quadtree
        covariance = self.model.covariance

        ncombinations = quadtree.nleaves*(quadtree.nleaves+1)/2

        self.progress.setLabelText(text)
        self.progress.setMaximum(ncombinations)
        self.progress.setValue(0)

        @QtCore.pyqtSlot()
        def updateProgress():
            self.progress.setValue(covariance.finished_combinations)

        self.progress_timer = QtCore.QTimer()
        self.progress_timer.timeout.connect(updateProgress)
        self.progress_timer.start(250)

        self.progress.show()

    @QtCore.pyqtSlot()
    def processingFinished(self):
        self.progress.reset()
        if self.progress_timer is not None:
            self.progress_timer.stop()
            self.progress_timer = None

    def onSaveConfig(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter='YAML file *.yml (*.yml)', caption='Save scene YAML config')
        if not validateFilename(filename):
            return
        self.model.scene.saveConfig(filename)
        QtWidgets.QMessageBox.information(
            self, 'Scene config saved',
            'Scene config successfuly saved!'
            '<p style="font-family: monospace;">%s'
            '</p>' % filename)

    def onSaveScene(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Save scene')
        if not validateFilename(filename):
            return
        self.model.scene.save(filename)
        QtWidgets.QMessageBox.information(
            self, 'Scene saved',
            'Scene successfuly saved!'
            '<p style="font-family: monospace;">%s'
            '</p>' % filename)

    def onLoadConfig(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter='YAML file *.yml (*.yml)', caption='Load scene YAML config')
        if not validateFilename(filename):
            return
        self.sigLoadConfig.emit(filename)

    def onOpenScene(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Load kite scene')
        if not validateFilename(filename):
            return
        self.model.loadFile(filename)

    def onImportScene(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
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
        filename, flt = QtWidgets.QFileDialog.getSaveFileName(
            filter='GeoJSON (*.geojson *.json);;CSV File *.csv (*.csv)',
            caption='Export Quadtree')

        if not validateFilename(filename):
            return

        if flt == 'GeoJSON (*.geojson *.json)':
            if not filename.endswith('.geojson') and \
                    not filename.endswith('.json'):
                filename += '.geojson'
            self.model.quadtree.export_geojson(filename)

        elif flt == 'CSV File *.csv (*.csv)':
            if not filename.endswith('.csv'):
                filename += '.geojson'
            self.model.quadtree.export_csv(filename)
        else:
            raise ValueError('unknown filter')

    def onExportWeightMatrix(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter='Text File *.txt (*.txt)',
            caption='Export Covariance Weights',)
        if not validateFilename(filename):
            return
        self.sigExportWeightMatrix.emit(filename)

    def exit(self):
        pass


class KiteParameterTree(pg.parametertree.ParameterTree):
    pass


def spool(*args, **kwargs):
    spool_app = Spool(*args, **kwargs)
    spool_app.exec_()
    spool_app.quit()


__all__ = ['Spool', 'spool']

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    Spool(scene=sc)
