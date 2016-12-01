#!/usr/bin/python2
# import logging
from PySide import QtGui, QtCore
from .tab_scene import QKiteScene
from .tab_quadtree import QKiteQuadtree
from .tab_covariance import QKiteCovariance
from os import path
from utils_qt import loadUi
import pyqtgraph as pg
from ..meta import Subject


class SpoolMainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.loadUi()

        self.actionSave_YAML.triggered.connect(self.onSaveYaml)
        self.actionSave_Scene.triggered.connect(self.onSaveData)

        self.scene = None
        self.views = []
        self.ptree = QKiteParameterTree(showHeader=True)

        self.splitter.insertWidget(0, self.ptree)

        self.loadingModule = Subject()

    def loadUi(self):
        ui_file = path.join(path.dirname(path.realpath(__file__)),
                            'ui/spool.ui')
        loadUi(ui_file, baseinstance=self)
        return

    def addScene(self, scene):
        self.scene = scene

        for v in [QKiteScene, QKiteQuadtree, QKiteCovariance]:
            self.addView(v)

    def addView(self, view):
        view = view(self)
        self.loadingModule._notify(view.title)

        self.tabs.addTab(view, view.title)

        if hasattr(view, 'parameters'):
            for parameter in view.parameters:
                self.ptree.addParameters(parameter)
        self.views.append(view)

    def onSaveYaml(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='*.yml', caption='Save scene YAML config')
        if filename == '':
            return
        self.scene.save_config(filename)

    def onSaveData(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            filter='*', caption='Save scene')
        if filename == '':
            return
        self.scene.save(filename)

    def exit(self):
        pass


class QKiteParameterTree(pg.parametertree.ParameterTree):
    pass


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

        updateSplashMessage('Scene')
        splash.show()
        self.processEvents()

        self.aboutToQuit.connect(self.deleteLater)

        self.spool_win = SpoolMainWindow()
        self.spool_win.loadingModule.subscribe(updateSplashMessage)

        self.spool_win.actionExit.triggered.connect(self.exit)

        if scene is not None:
            self.addScene(scene)

        self.spool_win.show()
        splash.finish(self.spool_win)
        self.exec_()

    def addScene(self, scene):
        return self.spool_win.addScene(scene)

    def __del__(self):
        pass


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
