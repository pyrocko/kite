#!/bin/python

from PySide.QtCore import *
from PySide.QtGui import *
import sys

class QSceneDisplacementPlot2D(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

class QSceneDisplacementControl(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

class QSceneDisplacement(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.exec()
