import numpy as num
from PyQt5 import QtCore, QtWidgets

from kite.qt_utils import loadUi

from .util import get_resource

km = 1e3


class ExtentDialog(QtWidgets.QDialog):
    def __init__(self, sandbox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loadUi(get_resource("dialog_extent.ui"), self)
        self.setSizeGripEnabled(False)

        self.move(
            self.parent().window().mapToGlobal(self.parent().window().rect().center())
            - self.mapToGlobal(self.rect().center())
        )

        self.sandbox = sandbox
        model = self.sandbox.model
        dE, dN = model.frame.dEmeter, model.frame.dNmeter

        def getKm(px, dp):
            return "%.2f km " % (dp * px / km)

        self.spinEastPx.valueChanged.connect(
            lambda px: self.eastKm.setText(getKm(px, dE))
        )
        self.spinNorthPx.valueChanged.connect(
            lambda px: self.northKm.setText(getKm(px, dN))
        )

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
        self.sandbox.model.setExtent(self.spinEastPx.value(), self.spinNorthPx.value())
        self.setValues(self.spinEastPx.value(), self.spinEastPx.value())
        self.setValues()


class LosDialog(QtWidgets.QDialog):
    def __init__(self, sandbox, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource("dialog_los.ui"), self)
        self.setSizeGripEnabled(False)

        self.move(
            self.parent().window().mapToGlobal(self.parent().window().rect().center())
            - self.mapToGlobal(self.rect().center())
        )

        self.sandbox = sandbox
        model = self.sandbox.model
        self.applyButton.released.connect(self.updateValues)
        self.okButton.released.connect(self.updateValues)
        self.okButton.released.connect(self.close)

        self.setValues()

    def setValues(self):
        model = self.sandbox.model
        phi = num.deg2rad(self.spinlos_phi.value())
        theta = num.deg2rad(self.spinlos_theta.value())

    @QtCore.pyqtSlot()
    def updateValues(self):
        print("updated los!")
        self.sandbox.model.setLOS(self.spinlos_phi.value(), self.spinlos_theta.value())
        self.setValues()
