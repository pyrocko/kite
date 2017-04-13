from PySide import QtCore, QtGui
import numpy as num
import os

import pyqtgraph as pg

from .common import RectangularSourceROI, PointSourceROI, SourceDelegate
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

    display_backend = 'pyrocko'
    display_name = 'RectangularSource'

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
    {idx}. {delegate.display_name}
    <span style="color: #616161;">
        ({delegate.display_backend})
    </span>
</span>
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
</tr></table>
'''
        return item.format(idx=self.index.row()+1,
                           delegate=self,
                           source=self.source)


class PyrockoMomentTensorDialog(PyrockoSourceDialog):

    def __init__(self, *args, **kwargs):
        PyrockoSourceDialog.__init__(
            self, ui_file='pyrocko_moment_tensor.ui', *args, **kwargs)

        mt = ['mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']

        exponent = float(
            num.log10(
                num.mean([self.__getattribute__(n).value() for n in mt])))
        exponent = exponent if exponent > 0 else 1
        self.exponent.setValue(exponent)

        def valueFromTextExp(sb, text):
            print 'text: ', text
            return float(text)**self.exponent.value()

        def textFromValueExp(sb, value):
            print 'value: ', value
            return '%.2f' % (value / 10**exponent)

        for spinbox_name in mt:
            spin = self.__getattribute__(spinbox_name)
            spin.__setattr__('valueFromText', valueFromTextExp)
            spin.__setattr__('textFromValue', textFromValueExp)


class PyrockoMomentTensorDelegate(SourceDelegate):

    __represents__ = 'PyrockoMomentTensor'

    display_backend = 'pyrocko'
    display_name = 'MomentTensor'

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
    {idx}. {delegate.display_name}
    <span style="color: #616161;">
        ({delegate.display_backend})
    </span>
</span>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>mnn:</td><td style="align: justify;">{source.mnn:.3e}</td><td>Nm</td>
</tr><tr>
    <td>mee:</td><td style="align: justify;">{source.mee:.3e}</td><td>Nm</td>
</tr><tr>
    <td>mdd:</td><td style="align: justify;">{source.mdd:.3e}</td><td>Nm</td>
</tr><tr>
    <td>mne:</td><td style="align: justify;">{source.mne:.3e}</td><td>Nm</td>
</tr><tr>
    <td>mnd:</td><td style="align: justify;">{source.mnd:.3e}</td><td>Nm</td>
</tr><tr>
    <td>med:</td><td style="align: justify;">{source.med:.3e}</td><td>Nm</td>
</tr>
</table>
'''
        return item.format(idx=self.index.row()+1,
                           delegate=self,
                           source=self.source)


class PyrockoDoubleCoupleDialog(PyrockoSourceDialog):

    def __init__(self, *args, **kwargs):
        PyrockoSourceDialog.__init__(
            self, ui_file='pyrocko_double_couple.ui', *args, **kwargs)


class DoubleCoupleROI(PointSourceROI):

    def __init__(self, *args, **kwargs):
        PointSourceROI.__init__(self, *args, **kwargs)
        self.addRotateHandle([.5, 1.], [0.5, 0.5])
        self.updateROIPosition()

    @QtCore.Slot()
    def setSourceParametersFromROI(self):
        angle = self.angle()
        strike = float(-angle) % 360
        parameters = {
            'easting':
                float(
                    self.pos().x() + num.sin(strike*d2r) * self.size().x()/2),
            'northing':
                float(
                    self.pos().y() + num.cos(strike*d2r) * self.size().y()/2),
            'strike': strike
            }

        parameters = {
            'easting': float(self.pos().x()),
            'northing': float(self.pos().y()),
            'strike': strike
        }
        self.newSourceParameters.emit(parameters)

    @QtCore.Slot()
    def updateROIPosition(self):
        source = self.source
        self.setPos(
            QtCore.QPointF(
                source.easting
                - num.sin(source.strike*d2r) * self.size().x()/2,
                source.northing
                - num.cos(source.strike*d2r) * self.size().y()/2),
            finish=False)
        self.setPos(QtCore.QPointF(source.easting, source.northing),
                    finish=False)
        self.setAngle(-source.strike, finish=False)

    def paint(self, p, opt, widget):
        return pg.ROI.paint(self, p, opt, widget)


class PyrockoDoubleCoupleDelegate(SourceDelegate):

    __represents__ = 'PyrockoDoubleCouple'

    display_backend = 'pyrocko'
    display_name = 'DoubleCouple'

    parameters = ['easting', 'northing', 'depth', 'store_dir',
                  'strike', 'dip', 'rake', 'magnitude']
    ro_parameters = []
    ROIWidget = DoubleCoupleROI

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
    {idx}. {delegate.display_name}
    <span style="color: #616161;">
        ({delegate.display_backend})
    </span>
</span>
<table style="color: #616161; font-size: small;">
<tr>
    <td>Depth:</td><td>{source.depth:.2f} m</td>
</tr><tr>
    <td>Strike:</td><td>{source.strike:.2f}&deg;</td>
</tr><tr>
    <td>Dip:</td><td>{source.dip:.2f}&deg;</td>
</tr><tr>
    <td>Rake:</td><td>{source.rake:.2f}&deg;</td>
</tr><tr>
    <td>M<sub>0</sub>:</td><td>{source.moment:.2e}</td>
</tr><tr style="font-weight: bold;">
    <td>M<sub>W</sub>:</td><td>{source.magnitude:.2f}</td>
</tr>
</table>
'''
        return item.format(idx=self.index.row()+1,
                           delegate=self,
                           source=self.source)
