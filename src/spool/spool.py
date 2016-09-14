#!/bin/python

from PySide import QtGui
from PySide import QtCore

import scene_qtgraph
from os import path


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

        uifile = QtCore.QFile(path.dirname(path.realpath(__file__))
                              + '/ui/spool.ui')
        uifile.open(QtCore.QFile.ReadOnly)

        ui = QtUiTools.QUiLoader().load(uifile, parent)
        return ui


class Spool(QtGui.QApplication):
    def __init__(self, scene=None):
        QtGui.QApplication.__init__(self, ['KiteSpool'])

        self.spool_win = SpoolMainWindow()
        if scene is not None:
            self.addScene(scene)

        self.exec_()

    def addScene(self, scene):
        return self.spool_win.addScene(scene)


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
