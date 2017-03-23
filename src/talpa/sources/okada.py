from PySide import QtCore
import pyqtgraph as pg
import numpy as num

d2r = num.pi / 180.


class OkadaSourceROI(pg.ROI):

    newSourceParameters = QtCore.Signal(object)

    pen_outline = pg.mkPen((46, 125, 50, 180), width=1.25)
    pen_handle = pg.mkPen(((38, 50, 56, 180)), width=1.25)
    pen_highlight = pg.mkPen((52, 175, 60), width=2.5)

    def __init__(self, okada_model):
        self.model = okada_model
        self.source = self.model.source
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

        self.model.sourceParametersChanged.connect(self.updatePosition)
        self.sigRegionChangeFinished.connect(self.updateSourceParameters)

    @QtCore.Slot()
    def updateSourceParameters(self):
        strike = float(180. - self.angle())
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
    def updatePosition(self):
        self.setPos(
            pg.Point(self.source.outline()[0]),
            finish=False)
        self.setSize(
            pg.Point(self.source.width, self.source.length),
            finish=False)
        self.setAngle(
            180. - self.source.strike,
            finish=False)

    def _makePen(self):
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return self.pen_highlight
        else:
            return self.pen


class OkadaSourceModel(QtCore.QObject):

    __represents__ = 'OkadaSource'

    sourceParametersChanged = QtCore.Signal()
    highlightROI = QtCore.Signal()
    ROISelected = QtCore.Signal(QtCore.QModelIndex)

    def __init__(self, model, source, index):
        QtCore.QObject.__init__(self)
        self.source = source
        self.model = model
        self.index = index

    def getROIItem(self):
        src = OkadaSourceROI(self)
        src.newSourceParameters.connect(self.updateParameters)
        src.sigHoverEvent.connect(
            lambda: self.ROISelected.emit(self.index))
        return src

    @QtCore.Slot(object)
    def updateParameters(self, parameters):
        for param, value in parameters.iteritems():
            self.source.__setattr__(param, value)
        self.source.parametersUpdated()
        self.sourceParametersChanged.emit()

    def formatListItem(self):
        item = '''{0}. <i>OkadaSource</i>
<table style="color: #616161; font-size: small;">
<tr>
    <td style="width: 100px;">Depth:</td><td>{source.depth:.2f} m</td>
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
</table>
'''
        return item.format(self.index.row()+1, source=self.source)
