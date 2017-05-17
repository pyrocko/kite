from PySide import QtCore, QtGui
import pyqtgraph as pg
import numpy as num
from ..common import get_resource
from kite.qt_utils import loadUi

d2r = num.pi / 180.
r2d = 180. / num.pi


class RectangularSourceROI(pg.ROI):

    newSourceParameters = QtCore.Signal(object)

    pen_outline = pg.mkPen((46, 125, 50, 180), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 180)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, delegate):
        self.delegate = delegate
        self.source = self.delegate.source
        source = self.source

        pg.ROI.__init__(
            self,
            pos=pg.Point(source.outline()[0]),
            size=pg.Point(source.width, source.length),
            angle=180. - source.strike,
            invertible=False,
            pen=self.pen_outline)
        self.handlePen = self.pen_handle

        self.addScaleRotateHandle([0, 0], [0, 1])
        self.addScaleRotateHandle([0, 1], [0, 0])
        self.addScaleHandle([1, .5], [0, .5],
                            lockAspect=False)

        for h in self.handles:
            h['item'].sigClicked.connect(self.sigRegionChangeStarted.emit)

        self.delegate.sourceParametersChanged.connect(
            self.updateROIPosition)
        self.sigRegionChangeFinished.connect(
            self.setSourceParametersFromROI)

        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.sigClicked.connect(self.showEditingDialog)

    @QtCore.Slot()
    def setSourceParametersFromROI(self):
        strike = float((180. - self.angle()) % 360)
        parameters = {
            'strike': strike,
            'width': float(self.size().x()),
            'length': float(self.size().y()),
            'easting': float(
                self.pos().x() - num.sin(strike * d2r) * self.size().y()/2),
            'northing': float(
                self.pos().y() - num.cos(strike * d2r) * self.size().y()/2)
            }
        self.newSourceParameters.emit(parameters)

    @QtCore.Slot()
    def updateROIPosition(self):
        source = self.source
        width = source.width

        self.setPos(
            pg.Point(source.outline()[0]),
            finish=False)
        self.setSize(
            pg.Point(width, source.length),
            finish=False)
        self.setAngle(
            180. - source.strike,
            finish=False)

    @QtCore.Slot()
    def highlightROI(self, highlight):
        self.setMouseHover(highlight)

    @QtCore.Slot()
    def showEditingDialog(self, *args):
        self.delegate.getEditingDialog().show()

    def _makePen(self):
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return self.pen_highlight
        else:
            return self.pen


class PointSourceROI(pg.EllipseROI):

    newSourceParameters = QtCore.Signal(object)

    pen_outline = pg.mkPen((46, 125, 50, 180), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 180)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, delegate):
        self.delegate = delegate
        self.source = self.delegate.source
        source = self.source

        size = 3000.
        pg.EllipseROI.__init__(
            self,
            pos=(source.easting - size/2, source.northing - size/2),
            size=size,
            invertible=False,
            pen=self.pen_outline)
        self.handlePen = self.pen_handle
        self.aspectLocked = True
        self.handles = []
        # self.setFlag(self.ItemIgnoresTransformations)
        self.setScale(1.)

        self.delegate.sourceParametersChanged.connect(
            self.updateROIPosition)
        self.sigRegionChangeFinished.connect(
            self.setSourceParametersFromROI)

        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        self.sigClicked.connect(self.showEditingDialog)

    @QtCore.Slot()
    def setSourceParametersFromROI(self):
        parameters = {
            'easting': float(self.pos().x() + self.size().x()/2),
            'northing': float(self.pos().y() + self.size().y()/2)
            }
        self.newSourceParameters.emit(parameters)

    @QtCore.Slot()
    def updateROIPosition(self):
        source = self.source
        self.setPos(
            pg.Point((source.easting - self.size().x()/2,
                      source.northing - self.size().x()/2)),
            finish=False)

    @QtCore.Slot()
    def highlightROI(self, highlight):
        self.setMouseHover(highlight)

    @QtCore.Slot()
    def showEditingDialog(self, *args):
        self.delegate.getEditingDialog().show()


class SourceDelegate(QtCore.QObject):

    __represents__ = 'SourceToImplement'

    sourceParametersChanged = QtCore.Signal()
    highlightROI = QtCore.Signal(bool)

    parameters = ['List', 'of', 'parameters', 'from', 'kite.source']
    ro_parameters = ['Read-Only', 'parameters']

    ROIWidget = None  # For use in pyqtgraph
    EditDialog = None  # QDialog to edit the source

    def __init__(self, model, source, index):
        QtCore.QObject.__init__(self)
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
        return '<b>Richtext</b> Source details'

    def getROIItem(self):
        src = self.ROIWidget(self)

        src.newSourceParameters.connect(
            self.updateModelParameters)
        src.sigRegionChangeStarted.connect(
            self.highlightItem)
        src.sigHoverEvent.connect(
            self.highlightItem)
        self.highlightROI.connect(
            src.highlightROI)

        self.rois.append(src)
        return src

    @QtCore.Slot(object)
    def highlightItem(self):
        if self.model.selection_model is not None:
            self.model.selection_model.setCurrentIndex(
                self.index, QtGui.QItemSelectionModel.SelectCurrent)

    @QtCore.Slot()
    def selectionChanged(self, idx):
        if self.index.row() == idx.row()\
          and self.index.column() == idx.column():
            self.highlightROI.emit(True)
        else:
            self.highlightROI.emit(False)

    @QtCore.Slot()
    def setSelectionModel(self):
        self.model.selection_model.currentChanged.connect(
            self.selectionChanged)

    @QtCore.Slot(object)
    def updateModelParameters(self, parameters):
        self.model.setItemData(self.index, parameters)

    def setSourceParameters(self, parameters):
        for param, value in parameters.iteritems():
            self.source.__setattr__(param, value)
        self.source.parametersUpdated()
        self.sourceParametersChanged.emit()

    def getSourceParameters(self):
        params = {}
        for param in self.parameters + self.ro_parameters:
            params[param] = self.source.__getattribute__(param)
        return params


class SourceEditDialog(QtGui.QDialog):

    def __init__(self, delegate, ui_file, *args, **kwargs):
        QtGui.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource(ui_file), self)
        self.delegate = delegate

        self.delegate.sourceParametersChanged.connect(
            self.getSourceParameters)
        self.applyButton.released.connect(
            self.setSourceParameters)
        self.okButton.released.connect(
            self.setSourceParameters)
        self.okButton.released.connect(
            self.close)

    @QtCore.Slot()
    def getSourceParameters(self):
        for param, value in self.delegate.getSourceParameters().iteritems():
            self.__getattribute__(param).setValue(value)

    @QtCore.Slot()
    def setSourceParameters(self):
        params = {}
        for param in self.delegate.parameters:
            params[param] = self.__getattribute__(param).value()
        self.delegate.updateModelParameters(params)
