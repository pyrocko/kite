import numpy as num

from kite.sources import OkadaSource

from .base import RectangularSourceROI, SourceDelegate, SourceEditDialog

d2r = num.pi / 180.0
r2d = 180.0 / num.pi


class OkadaSourceDelegate(SourceDelegate):

    __represents__ = "OkadaSource"

    display_backend = "Okada"
    display_name = "OkadaSource"

    parameters = [
        "easting",
        "northing",
        "width",
        "length",
        "depth",
        "slip",
        "opening",
        "strike",
        "dip",
        "rake",
        "nu",
    ]
    ro_parameters = ["seismic_moment", "moment_magnitude"]

    class OkadaDialog(SourceEditDialog):
        def __init__(self, delegate, *args, **kwargs):
            super().__init__(delegate, ui_file="okada_source.ui", *args, **kwargs)

            def setLabel(method, fmt, value, suffix=""):
                method(fmt.format(value) + suffix)

            self.moment_magnitude.setValue = lambda v: setLabel(
                self.moment_magnitude.setText, "{:.2f}", v
            )
            self.seismic_moment.setValue = lambda v: setLabel(
                self.seismic_moment.setText, "{:.2e}", v, " Nm"
            )

            self.getSourceParameters()

    ROIWidget = RectangularSourceROI
    EditDialog = OkadaDialog

    @staticmethod
    def getRepresentedSource(sandbox):
        length = 5000.0
        return OkadaSource(
            length=length, width=15.0 * length**0.66, strike=45.0, rake=0, slip=2
        )

    def formatListItem(self):
        item = """
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
"""
        return item.format(idx=self.index.row() + 1, delegate=self, source=self.source)
