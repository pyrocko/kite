#!/usr/bin/python
from PySide import QtGui, QtCore
from .tab_scene import QKiteScene
from .tab_quadtree import QKiteQuadtree
from .tab_covariance import QKiteCovariance
from .utils_qt import loadUi
from ..meta import Subject
from ..scene import Scene

from os import path
import os
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
    def __init__(self, scene=None):
        QtGui.QApplication.__init__(self, ['KiteSpool'])
        # self.setStyle('plastique')
        splash_img = QtGui.QPixmap(
            path.join(path.dirname(path.realpath(__file__)),
                      'ui/boxkite-sketch.jpg'))\
            .scaled(QtCore.QSize(400, 250), QtCore.Qt.KeepAspectRatio)
        splash = QtGui.QSplashScreen(splash_img,
                                     QtCore.Qt.WindowStaysOnTopHint)

        def updateSplashMessage(msg=''):
            splash.showMessage("Loading kite.%s ..." % msg.title(),
                               QtCore.Qt.AlignBottom)

        splash.show()
        self.processEvents()
        updateSplashMessage('Scene')

        self.aboutToQuit.connect(self.deleteLater)

        self.spool_win = SpoolMainWindow()
        self.spool_win.loadingModule.subscribe(updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)

        if scene is not None:
            self.addScene(scene)

        splash.finish(self.spool_win)
        self.spool_win.show()
        self.exec_()

    def addScene(self, scene):
        return self.spool_win.addScene(scene)

    def __del__(self):
        pass


class SpoolMainWindow(QtGui.QMainWindow):
    evSceneChanged = Subject()

    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.loadUi()

        self.scene = None
        self.views = [QKiteScene, QKiteQuadtree, QKiteCovariance]

        self.ptree = QKiteParameterTree(showHeader=True)
        self.ptree.resize(500, 500)
        self.splitter.insertWidget(0, self.ptree)

        self.log_model = SceneLogModel(self)
        self.log = SceneLog(self)

        self.about = QtGui.QDialog()
        about_ui = path.join(path.dirname(path.realpath(__file__)),
                             'ui/about.ui')
        loadUi(about_ui, baseinstance=self.about)

        self.actionSave_config.triggered.connect(
            self.onSaveConfig)
        self.actionSave_scene.triggered.connect(
            self.onSaveData)
        self.actionLoad_config.triggered.connect(
            self.onLoadConfig)
        self.actionExport_quadtree_CSV.triggered.connect(
            self.onExportQuadtreeCSV)
        self.actionLoad_scene.triggered.connect(
            self.onOpenScene)
        self.actionImport_scene.triggered.connect(
            self.onImportScene)
        self.actionAbout_Spool.triggered.connect(
            self.about.show)
        self.actionHelp.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl('http://pyrocko.org'))
        self.actionLog.triggered.connect(
            self.log.show)

        self.loadingModule = Subject()

    def loadUi(self):
        ui_file = path.join(path.dirname(path.realpath(__file__)),
                            'ui/spool.ui')
        loadUi(ui_file, baseinstance=self)
        return

    def addScene(self, scene):
        self.scene = scene
        self.log_model.attachScene(self.scene)

        for v in self.views:
            self.addView(v)

    def setScene(self, scene):
        self.scene = scene
        self.evSceneChanged.notify()

    def addView(self, view):
        view = view(self)
        self.loadingModule.notify(view.title)
        self.tabs.addTab(view, view.title)

        if hasattr(view, 'parameters'):
            for parameter in view.parameters:
                self.ptree.addParameters(parameter)

    def onSaveConfig(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='YAML file (*.yml)', caption='Save scene YAML config')
        if not validateFilename(filename):
            return
        self.scene.save_config(filename)

    def onSaveData(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='*', caption='Save scene')
        if not validateFilename(filename):
            return
        self.scene.save(filename)

    def onLoadConfig(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML file (*.yml)', caption='Load scene YAML config')
        if not validateFilename(filename):
            return
        self.scene.load_config(filename)

    def onExportQuadtreeCSV(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(
            filter='CSV File (*.csv)', caption='Export Quadtree CSV')
        if not validateFilename(filename):
            return
        self.scene.quadtree.export(filename)

    def onOpenScene(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='YAML, NumPy NPZ file (*.yml *.npz)',
            caption='Load kite scene')
        if not validateFilename(filename):
            return
        Scene.load(filename).spool()

    def onImportScene(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='Any file (*)',
            caption='Import scene to spool')
        if not validateFilename(filename):
            return
        Scene.import_data(filename).spool()

    def exit(self):
        pass


class QKiteParameterTree(pg.parametertree.ParameterTree):
    pass


class SceneLogModel(QtCore.QAbstractTableModel, logging.Handler):
    log_records = []

    def __init__(self, window, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)
        logging.Handler.__init__(self)

        self.window = window
        getPixmap = window.style().standardPixmap
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

    def attachScene(self, scene):
        self.scene = scene
        self.scene._log.addHandler(self)

    def detachScene(self, scene):
        self.scene._log.removeHandler(self)
        self.scene = None

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

    def emit(self, record):
        self.log_records.append(record)
        self.beginInsertRows(QtCore.QModelIndex(),
                             0, 0)
        self.endInsertRows()
        self.window.log.tableView.scrollToBottom()
        if record.levelno >= 30 and self.window.log.autoBox.isChecked():
            self.window.log.show()


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


__all__ = '''
Spool
'''.split()

if __name__ == '__main__':
    from kite.scene import SceneSynTest, Scene
    import sys
    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    Spool(scene=sc)
