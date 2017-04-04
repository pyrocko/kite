from PySide import QtCore, QtGui
import pyqtgraph as pg
import numpy as num

from ..common import get_resource
from kite.qt_utils import loadUi
from kite.sources import OkadaSource


d2r = num.pi / 180.
r2d = 180. / num.pi


class OkadaSourceROI(pg.ROI):

    newSourceParameters = QtCore.Signal(object)

    pen_outline = pg.mkPen((46, 125, 50, 180), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 180)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, okada_delegate):
        self.delegate = okada_delegate
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

        self.delegate.sourceParametersChanged.connect(self.updateROIPosition)
        self.sigRegionChangeFinished.connect(self.setSourceParametersFromROI)

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


class OkadaEditDialog(QtGui.QDialog):

    def __init__(self, delegate, *args, **kwargs):
        QtGui.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource('okada_source.ui'), self)
        self.delegate = delegate

        self.delegate.sourceParametersChanged.connect(self.getSourceParameters)
        self.applyButton.released.connect(self.setSourceParameters)
        self.okButton.released.connect(self.setSourceParameters)
        self.okButton.released.connect(self.close)

        def setLabel(method, fmt, value, suffix=''):
            method(fmt.format(value) + suffix)

        self.moment_magnitude.setValue = lambda v: setLabel(
            self.moment_magnitude.setText, '{:.2f}', v)
        self.seismic_moment.setValue = lambda v: setLabel(
            self.seismic_moment.setText, '{:.2e}', v, ' Nm')

        self.getSourceParameters()

    @QtCore.Slot()
    def getSourceParameters(self):
        for param, value in self.delegate.getSourceParameters().iteritems():
            self.__getattribute__(param).setValue(value)

    @QtCore.Slot()
    def setSourceParameters(self):
        params = {}
        for param in OkadaSourceDelegate.parameters:
            params[param] = float(self.__getattribute__(param).value())
        self.delegate.updateModelParameters(params)


class OkadaSourceDelegate(QtCore.QObject):

    __represents__ = 'OkadaSource'

    sourceParametersChanged = QtCore.Signal()
    highlightROI = QtCore.Signal(bool)

    parameters = ['easting', 'northing', 'width', 'length', 'depth',
                  'slip', 'opening', 'strike', 'dip', 'rake', 'nu']

    ro_parameters = ['seismic_moment', 'moment_magnitude']

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
        src = OkadaSource(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000,
            width=3000,
            length=5000,
            strike=45.,
            rake=0,
            slip=2,
            )
        return src

    def getROIItem(self):
        src = OkadaSourceROI(self)

        src.newSourceParameters.connect(self.updateModelParameters)

        src.sigRegionChangeStarted.connect(self.highlightItem)
        src.sigHoverEvent.connect(self.highlightItem)
        self.highlightROI.connect(src.highlightROI)

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

    def getEditingDialog(self):
        if self.editing_dialog is None:
            self.editing_dialog = OkadaEditDialog(self)
        return self.editing_dialog

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

    def formatListItem(self):
        item = '''{0}. <i>OkadaSource</i>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>Width:</td><td>{source.width:.2f} m</td>
</tr><tr>
    <td>Length:</td><td>{source.length:.2f} m</td>
</tr><tr>
    <td>Slip:</td><td>{source.slip:.2f} m</td>
</tr><tr>
    <td>Strike:</td><td>{source.strike:.2f}&deg;</td>
</tr><tr>
    <td>Dip:</td><td>{source.dip:.2f}&deg;</td>
</tr><tr>
    <td>Rake:</td><td>{source.rake:.2f}&deg;</td>
</tr>
<tr>
    <td>M<sub>W</sub>:</td><td>{source.moment_magnitude:.2f}</td>
</tr>
</table>
'''
        return item.format(self.index.row()+1, source=self.source)
