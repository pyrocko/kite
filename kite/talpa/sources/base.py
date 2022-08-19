import numpy as num
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from kite.qt_utils import loadUi

from ..util import get_resource

d2r = num.pi / 180.0
r2d = 180.0 / num.pi


class RectangularSourceROI(pg.ROI):

    newSourceParameters = QtCore.pyqtSignal(object)

    pen_outline = pg.mkPen((46, 125, 50, 100), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 100)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, delegate):
        self.delegate = delegate
        self.source = self.delegate.source
        source = self.source

        pg.ROI.__init__(
            self,
            pos=pg.Point(source.outline()[1]),
            size=pg.Point(source.width, source.length),
            angle=-source.strike,
            invertible=False,
            pen=self.pen_outline,
        )
        self.handlePen = self.pen_handle

        self.addScaleRotateHandle([0, 0], [0, 1])
        self.addScaleRotateHandle([0, 1], [0, 0])
        self.addScaleHandle([1, 0.5], [0, 0.5], lockAspect=False)

        for h in self.handles:
            h["item"].sigClicked.connect(self.sigRegionChangeStarted.emit)

        self.delegate.sourceParametersChanged.connect(self.updateROIPosition)
        self.sigRegionChangeFinished.connect(self.setSourceParametersFromROI)

        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.sigClicked.connect(self.showEditingDialog)

    @QtCore.pyqtSlot()
    def setSourceParametersFromROI(self):

        strike = float((-self.angle()) % 360)
        width = float(self.size().x())
        length = float(self.size().y())

        northing = float(self.pos().y() + num.cos(strike * d2r) * length / 2)
        easting = float(self.pos().x() + num.sin(strike * d2r) * length / 2)

        north_shift, east_shift = self.source.getSandboxOffset()
        easting -= east_shift
        northing -= north_shift

        self.newSourceParameters.emit(
            dict(
                strike=strike,
                width=width,
                length=length,
                easting=easting,
                northing=northing,
            )
        )

    @QtCore.pyqtSlot()
    def updateROIPosition(self):
        source = self.source
        width = source.width

        self.setPos(pg.Point(source.outline()[1]), finish=False)
        self.setSize(pg.Point(width, source.length), finish=False)
        self.setAngle(-source.strike, finish=False)

    @QtCore.pyqtSlot(bool)
    def highlightROI(self, highlight):
        self.setMouseHover(highlight)

    @QtCore.pyqtSlot()
    def showEditingDialog(self, *args):
        self.delegate.getEditingDialog().show()

    def _makePen(self):
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return self.pen_highlight
        else:
            return self.pen


class PointSourceROI(pg.EllipseROI):

    newSourceParameters = QtCore.pyqtSignal(object)

    pen_outline = pg.mkPen((46, 125, 50, 100), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 100)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, delegate):
        self.delegate = delegate
        self.source = self.delegate.source
        source = self.source

        size = 3000.0
        super().__init__(
            pos=(source.easting - size / 2, source.northing - size / 2),
            size=size,
            invertible=False,
            pen=self.pen_outline,
        )
        self.handlePen = self.pen_handle
        self.aspectLocked = True

        for h in self.handles:
            hi = h["item"]
            hi.disconnectROI(self)

        self.handles = []
        # self.setFlag(self.ItemIgnoresTransformations)
        self.setScale(1.0)

        self.delegate.sourceParametersChanged.connect(self.updateROIPosition)
        self.sigRegionChangeFinished.connect(self.setSourceParametersFromROI)

        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.sigClicked.connect(self.showEditingDialog)

    @QtCore.pyqtSlot()
    def setSourceParametersFromROI(self):
        north_shift, east_shift = self.source.getSandboxOffset()
        self.newSourceParameters.emit(
            dict(
                easting=float(self.pos().x() + self.size().x() / 2 - east_shift),
                northing=float(self.pos().y() + self.size().y() / 2 - north_shift),
            )
        )

    @QtCore.pyqtSlot()
    def updateROIPosition(self):
        source = self.source
        self.setPos(
            pg.Point(
                (
                    source.easting - self.size().x() / 2,
                    source.northing - self.size().x() / 2,
                )
            ),
            finish=False,
        )

    @QtCore.pyqtSlot(bool)
    def highlightROI(self, highlight):
        self.setMouseHover(highlight)

    @QtCore.pyqtSlot()
    def showEditingDialog(self, *args):
        self.delegate.getEditingDialog().show()

    def _makePen(self):
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return self.pen_highlight
        else:
            return self.pen


class SourceDelegate(QtCore.QObject):

    __represents__ = "SourceToImplement"

    sourceParametersChanged = QtCore.pyqtSignal()
    highlightROI = QtCore.pyqtSignal(bool)

    parameters = ["List", "of", "parameters", "from", "kite.source"]
    ro_parameters = ["Read-Only", "parameters"]

    ROIWidget = None  # For use in pyqtgraph
    EditDialog = None  # QDialog to edit the source

    def __init__(self, model, source, index):
        super().__init__()
        self.source = source
        self.model = model
        self.index = index
        self.rois = []

        self.editing_dialog = None

        if model.selection_model is not None:
            self.setSelectionModel()

        self.model.selectionModelChanged.connect(self.setSelectionModel)

    @staticmethod
    def getRepresentedSource(sandbox):
        raise NotImplementedError()

    def getEditingDialog(self):
        if self.editing_dialog is None:
            self.editing_dialog = self.EditDialog(self)
        return self.editing_dialog

    def formatListItem(self):
        raise NotImplementedError()
        return "<b>Richtext</b> Source details"

    def getROIItem(self):
        src = self.ROIWidget(self)

        src.newSourceParameters.connect(self.updateModelParameters)
        src.sigRegionChangeStarted.connect(self.highlightItem)
        src.sigHoverEvent.connect(self.highlightItem)
        self.highlightROI.connect(src.highlightROI)

        self.rois.append(src)
        return src

    @QtCore.pyqtSlot()
    def highlightItem(self):
        if self.model.selection_model is not None:
            self.model.selection_model.setCurrentIndex(
                self.index, QtCore.QItemSelectionModel.SelectCurrent
            )

    @QtCore.pyqtSlot()
    def emitHighlightROI(self):
        selected = self.model.selection_model.currentIndex()
        if (
            self.index.row() == selected.row()
            and self.index.column() == selected.column()
        ):
            self.highlightROI.emit(True)
        else:
            self.highlightROI.emit(False)

    @QtCore.pyqtSlot()
    def setSelectionModel(self):
        self.model.selection_model.currentChanged.connect(self.emitHighlightROI)

    @QtCore.pyqtSlot(object)
    def updateModelParameters(self, parameters):
        self.model.setItemData(self.index, parameters)

    def setSourceParameters(self, parameters):
        for param, value in parameters.items():
            self.source.__setattr__(param, value)
        self.source.parametersUpdated()
        self.sourceParametersChanged.emit()

    @QtCore.pyqtSlot()
    def getSourceParameters(self):
        params = {}
        for param in self.parameters + self.ro_parameters:
            params[param] = self.source.__getattribute__(param)
        return params


class SourceEditDialog(QtWidgets.QDialog):
    def __init__(self, delegate, ui_file, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource(ui_file), self)
        self.delegate = delegate

        self.delegate.sourceParametersChanged.connect(self.getSourceParameters)
        self.applyButton.released.connect(self.setSourceParameters)
        self.okButton.released.connect(self.setSourceParameters)
        self.okButton.released.connect(self.close)

    @QtCore.pyqtSlot()
    def getSourceParameters(self):
        for param, value in self.delegate.getSourceParameters().items():
            self.__getattribute__(param).setValue(value)

    @QtCore.pyqtSlot()
    def setSourceParameters(self):
        params = {}
        for param in self.delegate.parameters:
            params[param] = self.__getattribute__(param).value()
        self.delegate.updateModelParameters(params)
