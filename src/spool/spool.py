#!/bin/python

from PySide import QtGui
from PySide import QtCore

import qt_scene


class SpoolMainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)

        self.ui = self.loadUi(self)
        self.scenes = []

        self.ui.show()

    def addScene(self, scene):
        self.scenes.append(scene)
        self.ui.tabs.addTab(qt_scene.QKiteDisplacementTab(self.ui.tabs,
                                                          scene=scene),
                            'Displacement')
        self.ui.tabs.addTab(qt_scene.QKiteQuadtreeTab(self.ui.tabs,
                                                      scene=scene),
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

        self.spool_window = SpoolMainWindow()

if __name__ == '__main__':
    from kite.scene import SceneSynTest
    sc = SceneSynTest.createGauss()

    app = Spool()
    app.spool_window.addScene(sc)

    app.exec_()
