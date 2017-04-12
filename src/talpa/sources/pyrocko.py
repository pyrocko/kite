from PySide import QtCore, QtGui
import pyqtgraph as pg
import numpy as num
import os

from .common import RectangularSourceROI, SourceDelegate
from ..common import get_resource
from kite.qt_utils import loadUi
from kite.sources import (PyrockoRectangularSource,
                          PyrockoMomentTensor, PyrockoDoubleCouple)

from ..config import getConfig

d2r = num.pi / 180.
r2d = 180. / num.pi
config = getConfig()


class PyrockoSourceDialog(QtGui.QDialog):
    completer = QtGui.QCompleter()
    completer_model = QtGui.QFileSystemModel(completer)
    completer.setModel(completer_model)
    completer.setMaxVisibleItems(8)

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
        self.chooseStoreDirButton.released.connect(
            self.chooseStoreDir)

        self.completer_model.setRootPath('')
        self.completer.setParent(self.store_dir)
        self.store_dir.setCompleter(self.completer)

        self.store_dir.setValue = self.store_dir.setText
        self.store_dir.value = self.store_dir.text

        self.getSourceParameters()

    @QtCore.Slot()
    def chooseStoreDir(self):
        folder = QtGui.QFileDialog.getExistingDirectory(
            self, 'Open Pyrocko GF Store', os.getcwd())
        self.store_dir.setText(folder)

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


class PyrockoRectangularSourceEditDialog(PyrockoSourceDialog):

    def __init__(self, *args, **kwargs):
        PyrockoSourceDialog.__init__(
            self, ui_file='pyrocko_rectangular_source.ui', *args, **kwargs)


class PyrockoRectangularSourceDelegate(SourceDelegate):

    __represents__ = 'PyrockoRectangularSource'

    parameters = ['easting', 'northing', 'width', 'length', 'depth',
                  'slip', 'strike', 'dip', 'rake', 'store_dir',
                  'decimation_factor']
    ro_parameters = []

    ROIWidget = RectangularSourceROI

    @staticmethod
    def getRepresentedSource(sandbox):
        if not config.default_gf_dir:
            folder = QtGui.QFileDialog.getExistingDirectory(
                None, 'Open Pyrocko GF Store', os.getcwd())
        else:
            folder = config.default_gf_dir

        if not folder:
            return False

        length = 5000.
        src = PyrockoRectangularSource(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000,
            length=length,
            width=15. * length**.66,
            strike=45.,
            rake=0,
            slip=2,
            store_dir=folder,
            )
        return src

    def getEditingDialog(self):
        if self.editing_dialog is None:
            self.editing_dialog = PyrockoRectangularSourceEditDialog(self)
        return self.editing_dialog

    def formatListItem(self):
        item = '''
<span style="font-weight: bold; font-style: oblique">
    {0}. PyrockoRectangularSource</span>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>Width:</td><td>{source.width:.2f} m</td>
</tr><tr>
    <td>Length:</td><td>{source.length:.2f} m</td>
</tr><tr>
    <td>Strike:</td><td>{source.strike:.2f}&deg;</td>
</tr><tr>
    <td>Dip:</td><td>{source.dip:.2f}&deg;</td>
</tr><tr>
    <td>Rake:</td><td>{source.rake:.2f}&deg;</td>
</tr><tr style="font-weight: bold;">
    <td>Slip:</td><td>{source.slip:.2f} m</td>
</tr>
</table>
'''
        return item.format(self.index.row()+1, source=self.source)


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

        self.addTranslateHandle([.5*2.**-.5 + .5, .5*2.**-.5 + .5])

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


class PyrockoMomentTensorDialog(PyrockoSourceDialog):

    def __init__(self, *args, **kwargs):
        PyrockoSourceDialog.__init__(
            self, ui_file='pyrocko_moment_tensor.ui', *args, **kwargs)


class PyrockoDoubleCoupleDialog(PyrockoSourceDialog):

    def __init__(self, *args, **kwargs):
        PyrockoSourceDialog.__init__(
            self, ui_file='pyrocko_double_couple.ui', *args, **kwargs)


class PyrockoMomentTensorDelegate(SourceDelegate):

    __represents__ = 'PyrockoMomentTensor'

    parameters = ['easting', 'northing', 'depth', 'store_dir',
                  'mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']
    ro_parameters = []
    ROIWidget = PointSourceROI

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
        if not config.default_gf_dir:
            folder = QtGui.QFileDialog.getExistingDirectory(
                None, 'Open Pyrocko GF Store', os.getcwd())
        else:
            folder = config.default_gf_dir

        if not folder:
            return False

        src = PyrockoMomentTensor(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000.,
            store_dir=folder,
            )
        return src

    def getEditingDialog(self):
        if self.editing_dialog is None:
            self.editing_dialog = PyrockoMomentTensorDialog(self)
        return self.editing_dialog

    def formatListItem(self):
        item = '''
<span style="font-weight: bold; font-style: oblique">
    {0}. PyrockoMomentTensor</span>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>mnn:</td><td>{source.strike:.2f} Nm</td>
</tr><tr>
    <td>mee:</td><td>{source.mee:.2f} Nm</td>
</tr><tr>
    <td>mdd:</td><td>{source.mdd:.2f} Nm</td>
</tr><tr>
    <td>mne:</td><td>{source.mne:.2f} Nm</td>
</tr><tr>
    <td>mnd:</td><td>{source.mnd:.2f} Nm</td>
</tr><tr>
    <td>med:</td><td>{source.med:.2f} Nm</td>
</tr>
</table>
'''
        return item.format(self.index.row()+1, source=self.source)


class PyrockoDoubleCoupleDelegate(SourceDelegate):

    __represents__ = 'PyrockoDoubleCouple'

    parameters = ['easting', 'northing', 'depth', 'store_dir',
                  'strike', 'dip', 'rake', 'magnitude']
    ro_parameters = []
    ROIWidget = PointSourceROI

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
        if not config.default_gf_dir:
            folder = QtGui.QFileDialog.getExistingDirectory(
                None, 'Open Pyrocko GF Store', os.getcwd())
        else:
            folder = config.default_gf_dir

        if not folder:
            return False

        src = PyrockoDoubleCouple(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000.,
            store_dir=folder,
            )
        return src

    def getEditingDialog(self):
        if self.editing_dialog is None:
            self.editing_dialog = PyrockoDoubleCoupleDialog(self)
        return self.editing_dialog

    def formatListItem(self):
        item = '''
<span style="font-weight: bold; font-style: oblique">
    {0}. PyrockoDoubleCouple</span>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>Strike:</td><td>{source.strike:.2f}&deg;</td>
</tr><tr>
    <td>Dip:</td><td>{source.dip:.2f}&deg;</td>
</tr><tr>
    <td>Rake:</td><td>{source.rake:.2f}&deg;</td>
</tr><tr style="font-weight: bold;">
    <td>M<sub>W</sub>:</td><td>{source.magnitude:.2f}</td>
</tr>
</table>
'''
        return item.format(self.index.row()+1, source=self.source)
