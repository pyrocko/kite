from PyQt5 import QtWidgets, QtCore

from kite.qt_utils import loadUi
from .util import get_resource


class ExtentDialog(QtWidgets.QDialog):
    def __init__(self, sandbox, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource('dialog_extent.ui'), self)
        self.setSizeGripEnabled(False)

        self.move(
            self.parent().window().mapToGlobal(
                self.parent().window().rect().center()) -
            self.mapToGlobal(self.rect().center()))

        self.sandbox = sandbox
        model = self.sandbox.model
        dE, dN = model.frame.dE, model.frame.dN

        def getKm(px, dp):
            return '%.2f km ' % (dp * px * 1e-3)

        self.spinEastPx.valueChanged.connect(
            lambda px: self.eastKm.setText(getKm(px, dE)))
        self.spinNorthPx.valueChanged.connect(
            lambda px: self.northKm.setText(getKm(px, dN)))

        self.applyButton.released.connect(self.updateValues)
        self.okButton.released.connect(self.updateValues)
        self.okButton.released.connect(self.close)

        self.setValues()

    def setValues(self, east=None, north=None):
        model = self.sandbox.model

        east = model.config.extent_east if east is None else east
        north = model.config.extent_north if north is None else north

        self.spinEastPx.setValue(east)
        self.spinNorthPx.setValue(north)

    @QtCore.pyqtSlot()
    def updateValues(self):
        self.sandbox.model.setExtent(
            self.spinEastPx.value(),
            self.spinNorthPx.value())

        self.setValues(
            self.spinEastPx.value(),
            self.spinEastPx.value())
