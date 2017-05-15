from PySide import QtCore, QtGui
import numpy as num

from .common import RectangularSourceROI, SourceDelegate
from ..common import get_resource
from kite.qt_utils import loadUi
from kite.sources import OkadaSource

d2r = num.pi / 180.
r2d = 180. / num.pi


class OkadaSourceDelegate(SourceDelegate):

    __represents__ = 'OkadaSource'

    display_backend = 'Okada'
    display_name = 'OkadaSource'

    parameters = ['easting', 'northing', 'width', 'length', 'depth',
                  'slip', 'opening', 'strike', 'dip', 'rake', 'nu']
    ro_parameters = ['seismic_moment', 'moment_magnitude']

    class OkadaEditDialog(QtGui.QDialog):

        def __init__(self, delegate, *args, **kwargs):
            QtGui.QDialog.__init__(self, *args, **kwargs)
            loadUi(get_resource('okada_source.ui'), self)
            self.delegate = delegate

            self.delegate.sourceParametersChanged.connect(
                self.getSourceParameters)
            self.applyButton.released.connect(
                self.setSourceParameters)
            self.okButton.released.connect(
                self.setSourceParameters)
            self.okButton.released.connect(
                self.close)

            def setLabel(method, fmt, value, suffix=''):
                method(fmt.format(value) + suffix)

            self.moment_magnitude.setValue = lambda v: setLabel(
                self.moment_magnitude.setText, '{:.2f}', v)
            self.seismic_moment.setValue = lambda v: setLabel(
                self.seismic_moment.setText, '{:.2e}', v, ' Nm')

            self.getSourceParameters()

        @QtCore.Slot()
        def getSourceParameters(self):
            for param, value in\
              self.delegate.getSourceParameters().iteritems():
                self.__getattribute__(param).setValue(value)

        @QtCore.Slot()
        def setSourceParameters(self):
            params = {}
            for param in OkadaSourceDelegate.parameters:
                params[param] = float(self.__getattribute__(param).value())
            self.delegate.updateModelParameters(params)

    ROIWidget = RectangularSourceROI
    EditDialog = OkadaEditDialog

    @staticmethod
    def getRepresentedSource(sandbox):
        length = 5000.
        src = OkadaSource(
            easting=num.mean(sandbox.frame.E),
            northing=num.mean(sandbox.frame.N),
            depth=4000,
            length=length,
            width=15. * length**.66,
            strike=45.,
            rake=0,
            slip=2,
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
</tr><tr>
    <td>M<sub>0</sub>:</td><td>{source.seismic_moment:.2e}</td>
</tr><tr style="font-weight: bold;">
    <td>M<sub>W</sub>:</td><td>{source.moment_magnitude:.2f}</td>
</tr><tr style="font-weight: bold;">
    <td>Slip:</td><td>{source.slip:.2f} m</td>
</tr>
</table>
'''
        return item.format(idx=self.index.row()+1,
                           delegate=self,
                           source=self.source)
