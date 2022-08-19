#!/usr/bin/python
import math
import os.path as op
import sys
import time  # noqa
from datetime import datetime

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from kite.qt_utils import QRangeSlider, SceneLog, loadUi, validateFilename
from kite.scene import Scene
from kite.scene_stack import SceneStack

from .base import get_resource
from .scene_model import SceneModel
from .tab_aps import KiteAPS  # noqa
from .tab_covariance import KiteCovariance  # noqa
from .tab_quadtree import KiteQuadtree  # noqa
from .tab_scene import KiteScene


class Spool(QtWidgets.QApplication):
    def __init__(self, scene=None, import_file=None, load_file=None):
        super().__init__(["Spool"])
        # self.setStyle('plastique')
        splash_img = QtGui.QPixmap(get_resource("spool_splash.png")).scaled(
            QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio
        )
        self.splash = QtWidgets.QSplashScreen(
            splash_img, QtCore.Qt.WindowStaysOnTopHint
        )
        self.updateSplashMessage("Scene")
        self.splash.show()
        self.processEvents()

        self.spool_win = SpoolMainWindow()
        self.spool_win.sigLoadingModule.connect(self.updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)
        self.aboutToQuit.connect(self.spool_win.model.worker_thread.quit)

        if scene is not None:
            self.addScene(scene)
        elif import_file is not None:
            self.importScene(import_file)
        elif load_file is not None:
            self.loadScene(load_file)

        self.spool_win.show()
        self.splash.finish(self.spool_win)

    @QtCore.pyqtSlot(str)
    def updateSplashMessage(self, msg=""):
        self.splash.showMessage(f"Loading {msg.title()} ...", QtCore.Qt.AlignBottom)

    def addScene(self, scene):
        self.spool_win.addScene(scene)

    def importScene(self, filename):
        self.spool_win.model.importFile(filename)

    def loadScene(self, filename):
        self.spool_win.model.loadFile(filename)


class SpoolMainWindow(QtWidgets.QMainWindow):
    VIEWS = [KiteScene, KiteQuadtree, KiteCovariance, KiteAPS]

    sigImportFile = QtCore.pyqtSignal(str)
    sigLoadFile = QtCore.pyqtSignal(str)
    sigLoadConfig = QtCore.pyqtSignal(str)
    sigExportWeightMatrix = QtCore.pyqtSignal(str)
    sigLoadingModule = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        loadUi(get_resource("spool.ui"), baseinstance=self)

        self.views = []
        self.active_view = None

        self.ptree = KiteParameterTree(showHeader=False)
        self.ptree.setMinimumWidth(400)
        self.dock_ptree = QtWidgets.QDockWidget("Parameters", self)
        self.dock_ptree.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.dock_ptree.setWidget(self.ptree)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_ptree)

        self.tabs = QtWidgets.QTabWidget(self)
        self.dock_tabs = QtWidgets.QDockWidget(self)
        self.dock_tabs.setTitleBarWidget(QtWidgets.QWidget())
        self.dock_tabs.setWidget(self.tabs)
        self.dock_tabs.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_tabs)
        self.setCentralWidget(self.dock_tabs)

        self.model = SceneModel(self)
        self.model.sigSceneModelChanged.connect(self.buildViews)

        # Connecting signals
        self.sigLoadFile.connect(self.model.loadFile)
        self.sigImportFile.connect(self.model.importFile)
        self.sigLoadConfig.connect(self.model.loadConfig)
        self.sigExportWeightMatrix.connect(self.model.exportWeightMatrix)

        self.actionSave_config.triggered.connect(self.onSaveConfig)
        self.actionSave_scene.triggered.connect(self.onSaveScene)
        self.actionLoad_config.triggered.connect(self.onLoadConfig)
        self.actionLoad_scene.triggered.connect(self.onOpenScene)

        self.actionImport_scene.triggered.connect(self.onImportScene)
        self.actionExport_quadtree.triggered.connect(self.onExportQuadtree)
        self.actionExport_weights.triggered.connect(self.onExportWeightMatrix)

        self.actionAbout_Spool.triggered.connect(self.aboutDialog().show)
        self.actionAbout_Qt5.triggered.connect(
            lambda: QtWidgets.QMessageBox.aboutQt(self)
        )
        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://pyrocko.org"))
        )

        self.log = SceneLog(self, self.model)
        self.actionLog.triggered.connect(self.log.show)

        self.progress = QtWidgets.QProgressDialog("", None, 0, 0, self)
        self.progress.setValue(0)
        self.progress.closeEvent = lambda ev: ev.ignore()
        self.progress.setMinimumWidth(400)
        self.progress.reset()
        self.progress_timer = None

        self.state_hash = None

    def aboutDialog(self):
        self._about = QtWidgets.QDialog(self)
        loadUi(get_resource("about.ui"), baseinstance=self._about)
        return self._about

    def addScene(self, scene):
        self.model.setScene(scene)
        self.buildViews()
        self.state_hash = scene.get_state_hash()

    def buildViews(self):
        scene = self.model.getScene()

        title = scene.meta.filename or "Untitled"
        self.setWindowTitle(f"Spool - {title}")
        if scene is None or self.tabs.count() != 0:
            return
        for v in self.VIEWS:
            self.addView(v)
        self.model.sigProgressStarted.connect(
            self.progressStarted, type=QtCore.Qt.QueuedConnection
        )
        self.model.sigProgressFinished.connect(
            self.progressFinished, type=QtCore.Qt.QueuedConnection
        )

        self.tabs.currentChanged.connect(self.activateView)

        if isinstance(scene, SceneStack):
            self.addTimeSlider()

        self.activateView(0)

    def addView(self, view):
        self.sigLoadingModule.emit(view.title)
        view = view(self)
        self.tabs.addTab(view, view.title)

        if hasattr(view, "parameters"):
            for parameter in view.parameters:
                self.ptree.addParameters(parameter)

        self.views.append(view)

    @QtCore.pyqtSlot(int)
    def activateView(self, index):
        if self.active_view is not None:
            self.active_view.deactivateView()

        self.active_view = self.views[index]
        self.active_view.activateView()

    @QtCore.pyqtSlot(object)
    def progressStarted(self, args):
        if self.progress.isVisible():
            return

        nargs = len(args)

        text = args[0]
        maximum = 0 if nargs < 2 else args[1]
        progress_func = None if nargs < 3 else args[2]

        self.progress.setWindowTitle("Processing...")
        self.progress.setLabelText(text)
        self.progress.setMaximum(maximum)
        self.progress.setValue(0)

        @QtCore.pyqtSlot()
        def updateProgress():
            if progress_func:
                self.progress.setValue(progress_func())

        self.progress_timer = QtCore.QTimer()
        self.progress_timer.timeout.connect(updateProgress)
        self.progress_timer.start(250)

        self.progress.show()

    @QtCore.pyqtSlot()
    def progressFinished(self):
        self.progress.reset()
        if self.progress_timer is not None:
            self.progress_timer.stop()
            self.progress_timer = None

    def getSceneDirname(self):
        if self.model.scene.meta.filename:
            return op.dirname(self.model.scene.meta.filename)
        return None

    def onSaveConfig(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            directory=self.getSceneDirname(),
            filter="YAML file *.yml (*.yml)",
            caption="Save scene YAML config",
        )
        if not validateFilename(filename):
            return
        self.model.scene.saveConfig(filename)
        QtWidgets.QMessageBox.information(
            self,
            "Scene config saved",
            "Scene config successfully saved!"
            f'<p style="font-family: monospace;">{filename}'
            "</p>",
        )
        self.state_hash = self.model.scene.get_state_hash()

    def onSaveScene(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            directory=self.getSceneDirname(),
            filter="YAML *.yml and NumPy container *.npz (*.yml *.npz)",
            caption="Save scene",
        )
        if not validateFilename(filename):
            return
        self.model.scene.save(filename)
        QtWidgets.QMessageBox.information(
            self,
            "Scene saved",
            "Scene successfully saved!"
            '<p style="font-family: monospace;">%s'
            "</p>" % filename,
        )
        self.state_hash = self.model.scene.get_state_hash()

    def onLoadConfig(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter="YAML file *.yml (*.yml)", caption="Load scene YAML config"
        )
        if not validateFilename(filename):
            return
        self.sigLoadConfig.emit(filename)

    def onOpenScene(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            filter="YAML *.yml and NumPy container *.npz (*.yml *.npz)",
            caption="Load kite scene",
        )
        if not validateFilename(filename):
            return
        self.model.loadFile(filename)

    def onImportScene(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            filter="Supported Formats, *.grd, *.geo, *unw*, *.mat "
            "(*.grd,*.geo,*unw*,*.mat);;"
            "GMT5SAR Scene, *.grd (*.grd);;"
            "ISCE Scene, *unw* (*unw*);;Gamma Scene *.geo (*.geo);;"
            "Matlab Container, *.mat (*.mat);;Any File * (*)",
            caption="Import scene to spool",
        )
        if not validateFilename(filename):
            return
        self.sigImportFile.emit(filename)

    def onExportQuadtree(self):
        filename, flt = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            filter="GeoJSON (*.geojson *.json);;CSV File *.csv (*.csv)",
            caption="Export Quadtree",
        )

        if not validateFilename(filename):
            return

        if flt == "GeoJSON (*.geojson *.json)":
            if not filename.endswith(".geojson") and not filename.endswith(".json"):
                filename += ".geojson"
            self.model.quadtree.export_geojson(filename)

        elif flt == "CSV File *.csv (*.csv)":
            if not filename.endswith(".csv"):
                filename += ".geojson"
            self.model.quadtree.export_csv(filename)
        else:
            raise ValueError("unknown filter")

    def onExportWeightMatrix(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            filter="Text File *.txt (*.txt)",
            caption="Export Covariance Weights",
        )
        if not validateFilename(filename):
            return
        self.sigExportWeightMatrix.emit(filename)

    def addTimeSlider(self):
        stack = self.model.getScene()

        self.time_slider = QRangeSlider(self)
        self.time_slider.setMaximumHeight(50)

        slider_tmin = math.ceil(stack.tmin)
        slider_tmax = math.floor(stack.tmax)

        def datetime_formatter(value):
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d")

        self.time_slider.setMin(slider_tmin)
        self.time_slider.setMax(slider_tmax)
        self.time_slider.setRange(slider_tmin, slider_tmax)
        self.time_slider.setFormatter(datetime_formatter)

        @QtCore.pyqtSlot(int)
        def changeTimeRange():
            tmin, tmax = self.time_slider.getRange()
            stack.set_time_range(tmin, tmax)

        self.time_slider.startValueChanged.connect(changeTimeRange)
        self.time_slider.endValueChanged.connect(changeTimeRange)

        self.dock_time_slider = QtWidgets.QDockWidget(
            "Displacement time series - range control", self
        )
        self.dock_time_slider.setWidget(self.time_slider)
        self.dock_time_slider.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.dock_time_slider.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea
        )

        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_time_slider)

    def closeEvent(self, ev):
        if self.state_hash == self.model.scene.get_state_hash():
            return

        msg_box = QtWidgets.QMessageBox(parent=self, text="The scene has been modified")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel
        )
        msg_box.setInformativeText("Do you want to save your changes?")
        msg_box.setDefaultButton(QtWidgets.QMessageBox.Save)
        ret = msg_box.exec()

        if ret == QtWidgets.QMessageBox.Save:
            self.onSaveScene()
        elif ret == QtWidgets.QMessageBox.Cancel:
            ev.ignore()


class KiteParameterTree(pg.parametertree.ParameterTree):
    pass


def spool(*args, **kwargs):
    spool_app = Spool(*args, **kwargs)
    spool_app.exec_()
    spool_app.quit()


__all__ = ["Spool", "spool"]

if __name__ == "__main__":
    from kite.scene import SceneSynTest

    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    Spool(scene=sc)
