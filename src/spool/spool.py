#!/usr/bin/python2
# import logging
from PySide import QtGui, QtCore
from .tab_scene import QKiteSceneDock
from .tab_quadtree import QKiteQuadtreeDock
from .tab_covariance import QKiteCovarianceDock
from os import path
from utils_qt import loadUi
from ..meta import Subject


class SpoolMainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.loadUi()
        self.scenes = []
        self.loadingModule = Subject()

    def addScene(self, scene):
        self.scenes.append(scene)
        self.tabs.setMovable(True)
        self.loadingModule._notify('Scene.displacent')
        self.tabs.addTab(QKiteSceneDock(scene),
                         'Displacement')
        self.loadingModule._notify('Scene.quadtree')
        self.tabs.addTab(QKiteQuadtreeDock(scene.quadtree),
                         'Quadtree')
        self.loadingModule._notify('Scene.quadtree.covariance')
        self.tabs.addTab(QKiteCovarianceDock(scene.quadtree.covariance),
                         'Covariance')

    def exit(self):
        pass

    def loadUi(self):
        ui_file = path.join(path.dirname(path.realpath(__file__)),
                            'ui/spool.ui')
        loadUi(ui_file, baseinstance=self)
        return


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
