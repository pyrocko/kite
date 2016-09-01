#!/bin/python

from PySide import QtGui
from PySide import QtCore

import scene_qtgraph


class SpoolMainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)

        self.ui = self.loadUi(self)
        self.scenes = []

        self.ui.show()

    def addScene(self, scene):
        self.scenes.append(scene)
        self.ui.tabs.setMovable(True)
        self.ui.tabs.addTab(scene_qtgraph.QKiteDisplacementDock(scene),
                            'Displacement')
        self.ui.tabs.addTab(scene_qtgraph.QKiteQuadtreeDock(scene.quadtree),
                            'Quadtree')

    @staticmethod
    def loadUi(parent):
        from PySide import QtUiTools

        uifile = QtCore.QFile('data/spool.ui')
        uifile.open(QtCore.QFile.ReadOnly)

        ui = QtUiTools.QUiLoader().load(uifile, parent)
        return ui


class Spool(QtGui.QApplication):
    def __init__(self, *args, **kwargs):
        QtGui.QApplication.__init__(self, ['KiteSpool'], *args, **kwargs)

if __name__ == '__main__':
    from kite.scene import SceneSynTest, Scene
    import sys
    if len(sys.argv) > 1:
        sc = Scene.load(sys.argv[1])
    else:
        sc = SceneSynTest.createGauss()

    app = Spool()

    spool_win = SpoolMainWindow()
    spool_win.addScene(sc)

    app.exec_()
