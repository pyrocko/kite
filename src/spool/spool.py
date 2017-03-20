#!/usr/bin/python
from PySide import QtGui, QtCore
from .scene_proxy import QSceneProxy
from .tab_scene import QKiteScene
from .tab_quadtree import QKiteQuadtree  # noqa
from .tab_covariance import QKiteCovariance  # noqa
from .utils_qt import loadUi
from ..scene import Scene

from os import path
import os
import sys
import time  # noqa
import logging
import pyqtgraph as pg


def validateFilename(filename):
    filedir = path.dirname(filename)
    if filename == '' or filedir == '':
        return False
    if path.isdir(filename) or not os.access(filedir, os.W_OK):
        QtGui.QMessageBox.critical(None, 'Path Error',
                                   'Could not access file <b>%s</b>'
                                   % filename)
        return False
    return True


class Spool(QtGui.QApplication):
    def __init__(self, scene=None, import_data=None, load_file=None):
        QtGui.QApplication.__init__(self, ['spool'])
        # self.setStyle('plastique')
        splash_img = QtGui.QPixmap(
            path.join(path.dirname(path.realpath(__file__)),
                      'ui/spool_splash.png'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        self.splash = QtGui.QSplashScreen(splash_img,
                                          QtCore.Qt.WindowStaysOnTopHint)
        self.updateSplashMessage('Scene')
        self.splash.show()
        self.processEvents()

        self.spool_win = SpoolMainWindow()
        self.spool_win.sigLoadingModule.connect(self.updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)
        self.aboutToQuit.connect(self.spool_win.scene_proxy.worker_thread.quit)
        self.aboutToQuit.connect(self.spool_win.scene_proxy.deleteLater)
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
        self.spool_win.scene_proxy.importFile(filename)

    def loadScene(self, filename):
        self.spool_win.scene_proxy.loadFile(filename)

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

        self.views = [QKiteScene, QKiteQuadtree, QKiteCovariance]

        self.ptree = QKiteParameterTree(showHeader=True)
        self.ptree.resize(100, 100)
        self.splitter.insertWidget(0, self.ptree)

        self.scene_proxy = QSceneProxy()
        self.scene_proxy.sigSceneModelChanged.connect(self.buildViews)

        self.sigLoadFile.connect(self.scene_proxy.loadFile)
        self.sigImportFile.connect(self.scene_proxy.importFile)
        self.sigLoadConfig.connect(self.scene_proxy.loadConfig)
        self.sigExportWeightMatrix.connect(
            self.scene_proxy.exportWeightMatrix)

        self.log_model = SceneLogModel(self)
        self.log = SceneLog(self)

        self.actionSave_config.triggered.connect(
            self.onSaveConfig)
        self.actionSave_scene.triggered.connect(
            self.onSaveData)
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
            self.about.show)
        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))
        self.actionLog.triggered.connect(
            self.log.show)

        self.progress = QtGui.QProgressDialog('', None, 0, 0, self)
        self.progress.setValue(0)
        self.progress.closeEvent = lambda e: e.ignore()
        self.progress.setMinimumWidth(400)
        self.progress.setWindowTitle('processing...')
        self.scene_proxy.sigProcessingFinished.connect(self.progress.reset)

    @property
    def about(self):
        self._about = QtGui.QDialog()
        about_ui = path.join(path.dirname(path.realpath(__file__)),
                             'ui/about.ui')
        loadUi(about_ui, baseinstance=self._about)
        return self._about

    def loadUi(self):
        ui_file = path.join(path.dirname(path.realpath(__file__)),
                            'ui/spool.ui')
        loadUi(ui_file, baseinstance=self)
        return

    def addScene(self, scene):
        self.scene_proxy.setScene(scene)
        self.buildViews()

    def buildViews(self):
        title = self.scene_proxy.scene.meta.filename or 'Untitled'
        self.setWindowTitle('Spool - %s' % title)
        if self.scene_proxy.scene is None or self.tabs.count() != 0:
            return
        for v in self.views:
            self.addView(v)
        self.scene_proxy.sigProcessingStarted.connect(self.processingStarted)

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
        self.scene_proxy.scene.saveConfig(filename)

    def onSaveData(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML *.yml and NumPy container *.npz (*.yml *.npz)',
            caption='Save scene')
        if not validateFilename(filename):
            return
        self.scene_proxy.scene.save(filename)

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
        self.scene_proxy.loadFile(filename)

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
        self.scene_proxy.quadtree.export(filename)

    def onExportWeightMatrix(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='Text File *.txt (*.txt)',
            caption='Export Covariance Weights',)
        if not validateFilename(filename):
            return
        self.sigExportWeightMatrix.emit(filename)

    def exit(self):
        pass


class QKiteParameterTree(pg.parametertree.ParameterTree):
    pass


class SceneLogModel(QtCore.QAbstractTableModel, logging.Handler):
    log_records = []

    def __init__(self, spool, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)
        logging.Handler.__init__(self)

        self.spool = spool
        self.spool.scene_proxy.sigLogRecord.connect(self.newRecord)
        getPixmap = spool.style().standardPixmap
        qstyle = QtGui.QStyle

        self.levels = {
            50: [getPixmap(qstyle.SP_MessageBoxCritical), 'Critical'],
            40: [getPixmap(qstyle.SP_MessageBoxCritical), 'Error'],
            30: [getPixmap(qstyle.SP_MessageBoxWarning), 'Warning'],
            20: [getPixmap(qstyle.SP_MessageBoxInformation), 'Info'],
            10: [getPixmap(qstyle.SP_FileIcon), 'Debug'],
        }

        for i in self.levels.itervalues():
            i[0] = i[0].scaledToHeight(20)

    def data(self, idx, role):
        rec = self.log_records[idx.row()]

        if role == QtCore.Qt.DisplayRole:
            if idx.column() == 0:
                return int(rec.levelno)
            elif idx.column() == 1:
                return '%s:%s' % (rec.levelname, rec.name)
            elif idx.column() == 2:
                return rec.getMessage()

        elif role == QtCore.Qt.ItemDataRole:
            return idx.data()

        elif role == QtCore.Qt.DecorationRole:
            if idx.column() != 0:
                return
            log_lvl = self.levels[int(idx.data())]
            return log_lvl[0]

        elif role == QtCore.Qt.ToolTipRole:
            if idx.column() == 0:
                return rec.levelname
            elif idx.column() == 1:
                return '%s.%s' % (rec.module, rec.funcName)
            elif idx.column() == 2:
                return 'Line %d' % rec.lineno

    def rowCount(self, idx):
        return len(self.log_records)

    def columnCount(self, idx):
        return 3

    @QtCore.Slot(object)
    def newRecord(self, record):
        self.log_records.append(record)
        self.beginInsertRows(QtCore.QModelIndex(),
                             0, 0)
        self.endInsertRows()
        self.spool.log.tableView.scrollToBottom()
        if record.levelno >= 30 and self.spool.log.autoBox.isChecked():
            self.spool.log.show()


class SceneLog(QtGui.QDialog):

    class LogFilter(QtGui.QSortFilterProxyModel):
        def __init__(self, *args, **kwargs):
            QtGui.QSortFilterProxyModel.__init__(self, *args, **kwargs)
            self.level = 0

        def setLevel(self, level):
            self.level = level
            self.setFilterRegExp('%s' % self.level)

    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        log_ui = path.join(path.dirname(path.realpath(__file__)),
                           'ui/logging.ui')
        loadUi(log_ui, baseinstance=self)

        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        self.table_filter = self.LogFilter()
        self.table_filter.setFilterKeyColumn(0)
        self.table_filter.setDynamicSortFilter(True)
        self.table_filter.setSourceModel(parent.log_model)

        self.tableView.setModel(self.table_filter)

        self.tableView.setColumnWidth(0, 30)
        self.tableView.setColumnWidth(1, 200)

        self.filterBox.addItems(
            [l[1] for l in parent.log_model.levels.values()] + ['All'])
        self.filterBox.setCurrentIndex(0)

        def changeFilter():
            for k, v in parent.log_model.levels.iteritems():
                if v[1] == self.filterBox.currentText():
                    self.table_filter.setLevel(k)
                    return

            self.table_filter.setLevel(0)

        self.filterBox.currentIndexChanged.connect(changeFilter)


__all__ = ['Spool']

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    Spool(scene=sc)
