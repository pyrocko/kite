from PySide import QtGui, QtCore

from .common import get_resource
from ..qt_utils import loadUi


class ExtentDialog(QtGui.QDialog):
    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QDialog.__init__(self, *args, **kwargs)
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

    def setValues(self):
        model = self.sandbox.model

        self.spinEastPx.setValue(
            model.config.extent_east)
        self.spinNorthPx.setValue(
            model.config.extent_north)

    @QtCore.Slot()
    def updateValues(self):
        self.sandbox.model.setExtent(
            self.spinEastPx.value(),
            self.spinNorthPx.value())
        self.setValues()
