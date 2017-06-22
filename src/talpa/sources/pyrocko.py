from PySide import QtCore, QtGui
import numpy as num
import os

from .common import (RectangularSourceROI, PointSourceROI, SourceDelegate,
                     SourceEditDialog)
from kite.sources import (PyrockoRectangularSource,
                          PyrockoMomentTensor, PyrockoDoubleCouple,
                          PyrockoRingfaultSource)

from ..config import config

d2r = num.pi / 180.
r2d = 180. / num.pi


class PyrockoSourceDialog(SourceEditDialog):
    completer = QtGui.QCompleter()
    completer_model = QtGui.QFileSystemModel(completer)
    completer.setModel(completer_model)
    completer.setMaxVisibleItems(8)

    def __init__(self, delegate, ui_file, *args, **kwargs):
        SourceEditDialog.__init__(self, delegate, ui_file, *args, **kwargs)

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
        if folder != '':
            self.store_dir.setText(folder)


class PyrockoRectangularSourceDelegate(SourceDelegate):

    __represents__ = 'PyrockoRectangularSource'

    display_backend = 'pyrocko'
    display_name = 'RectangularSource'

    parameters = ['easting', 'northing', 'width', 'length', 'depth',
                  'slip', 'strike', 'dip', 'rake', 'store_dir',
                  'decimation_factor']
    ro_parameters = []

    class RectangularSourceDialog(PyrockoSourceDialog):

        def __init__(self, *args, **kwargs):
            PyrockoSourceDialog.__init__(
                self, ui_file='pyrocko_rectangular_source.ui', *args, **kwargs)

    ROIWidget = RectangularSourceROI
    EditDialog = RectangularSourceDialog

    @staticmethod
    def getRepresentedSource(sandbox):
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
            store_dir=config.default_gf_dir or '',
            )
        return src

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


class PyrockoMomentTensorDelegate(SourceDelegate):

    __represents__ = 'PyrockoMomentTensor'

    display_backend = 'pyrocko'
    display_name = 'MomentTensor'

    parameters = ['easting', 'northing', 'depth', 'store_dir',
                  'mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']
    ro_parameters = []

    class MomentTensorDialog(PyrockoSourceDialog):

        scaling_params = ['mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']

        def __init__(self, *args, **kwargs):
            PyrockoSourceDialog.__init__(
                self, ui_file='pyrocko_moment_tensor.ui', *args, **kwargs)

        @QtCore.Slot()
        def setSourceParameters(self):
            params = {}
            scale = float('1e%d' % self.exponent.value())
            for param in self.delegate.parameters:
                params[param] = self.__getattribute__(param).value()
                if param in self.scaling_params:
                    params[param] = params[param] * scale
            self.delegate.updateModelParameters(params)

        @QtCore.Slot()
        def getSourceParameters(self):
            params = self.delegate.getSourceParameters()
            exponent = num.log10(
                num.max([v for k, v in params.iteritems()
                         if k in self.scaling_params]))
            scale = float('1e%d' % int(exponent))

            for param, value in params.iteritems():
                if param in self.scaling_params:
                    self.__getattribute__(param).setValue(value / scale)
                else:
                    self.__getattribute__(param).setValue(value)

    ROIWidget = PointSourceROI
    EditDialog = MomentTensorDialog

    @staticmethod
    def getRepresentedSource(sandbox):
        src = PyrockoMomentTensor(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000.,
            store_dir=config.default_gf_dir or '',
            )
        return src

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


class PyrockoDoubleCoupleDelegate(SourceDelegate):

    __represents__ = 'PyrockoDoubleCouple'

    display_backend = 'pyrocko'
    display_name = 'DoubleCouple'

    parameters = ['easting', 'northing', 'depth', 'store_dir',
                  'strike', 'dip', 'rake', 'magnitude']
    ro_parameters = []

    class DoubleCoupleDialog(PyrockoSourceDialog):

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
            vec_x, vec_y = self._vectorToCenter(strike)

            parameters = {
                'easting': float(self.pos().x() + vec_x),
                'northing': float(self.pos().y() + vec_y),
                'strike': strike
                }

            self.newSourceParameters.emit(parameters)

        @QtCore.Slot()
        def updateROIPosition(self):
            source = self.source
            vec_x, vec_y = self._vectorToCenter(source.strike)

            self.setAngle(-source.strike, finish=False)
            self.setPos(
                QtCore.QPointF(source.easting - vec_x,
                               source.northing - vec_y),
                finish=False)
            # self.setPos(QtCore.QPointF(source.easting, source.northing),
            #             finish=False)

        def _vectorToCenter(self, angle):
            rangle = angle * d2r

            sdx = self.size().x()/2
            sdy = self.size().y()/2

            return (sdx*num.sin(rangle) + sdy*num.cos(rangle),
                    sdx*num.cos(rangle) - sdy*num.sin(rangle))

    ROIWidget = DoubleCoupleROI
    EditDialog = DoubleCoupleDialog

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
        src = PyrockoDoubleCouple(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000.,
            store_dir=config.default_gf_dir or '',
            )
        return src

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


class PyrockoRingfaultDelegate(SourceDelegate):
    __represents__ = 'PyrockoRingfaultSource'

    display_backend = 'pyrocko'
    display_name = 'Ringfault'

    parameters = ['store_dir', 'easting', 'northing', 'depth', 'diameter',
                  'strike', 'dip', 'magnitude', 'npointsources']
    ro_parameters = []

    class RingfaultDialog(PyrockoSourceDialog):

        def __init__(self, *args, **kwargs):
            PyrockoSourceDialog.__init__(
                self, ui_file='pyrocko_ringfault.ui', *args, **kwargs)

    class RingfaultROI(PointSourceROI):

        def __init__(self, *args, **kwargs):
            PointSourceROI.__init__(self, *args, **kwargs)
            self.addScaleRotateHandle([.5, 1.], [.5, .5])
            self.updateROIPosition()

        @QtCore.Slot()
        def setSourceParametersFromROI(self):
            angle = self.angle()
            strike = float(-angle) % 360
            vec_x, vec_y = self._vectorToCenter(strike)

            parameters = {
                'easting': float(self.pos().x() + vec_x),
                'northing': float(self.pos().y() + vec_y),
                'diameter': self.size().y(),
                'strike': strike,
                }

            self.newSourceParameters.emit(parameters)

        @QtCore.Slot()
        def updateROIPosition(self):
            source = self.source

            self.setAngle(-source.strike, finish=False)
            self.setSize(source.diameter, finish=False)
            vec_x, vec_y = self._vectorToCenter(source.strike)
            self.setPos(
                QtCore.QPointF(source.easting - vec_x,
                               source.northing - vec_y), finish=False)
            # self.setPos(QtCore.QPointF(source.easting, source.northing),
            #             finish=False)

        def _vectorToCenter(self, angle):
            rangle = angle * d2r

            sdx = self.size().x()/2
            sdy = self.size().y()/2

            return (sdx*num.sin(rangle) + sdy*num.cos(rangle),
                    sdx*num.cos(rangle) - sdy*num.sin(rangle),)

    EditDialog = RingfaultDialog
    ROIWidget = RingfaultROI

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
        src = PyrockoRingfaultSource(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000.,
            diameter=10000.,
            store_dir=config.default_gf_dir or '',
            )
        return src

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
    <td>Diameter:</td><td>{source.diameter:.2f} m</td>
</tr><tr>
    <td>Strike:</td><td>{source.strike:.2f}&deg;</td>
</tr><tr>
    <td>Dip:</td><td>{source.dip:.2f}&deg;</td>
</tr><tr style="font-weight: bold;">
    <td>M<sub>W</sub>:</td><td>{source.magnitude:.2f}</td>
</tr>
</table>
'''
        return item.format(idx=self.index.row()+1,
                           delegate=self,
                           source=self.source)
