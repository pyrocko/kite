import numpy as num

from kite.sources import EllipsoidSource, PointCompoundSource

from .base import PointSourceROI, SourceDelegate, SourceEditDialog

d2r = num.pi / 180.0
r2d = 180.0 / num.pi


class EllipsoidSourceDelegate(SourceDelegate):

    __represents__ = "EllipsoidSource"

    display_backend = "Compound Model"
    display_name = "EllipsoidSource"

    parameters = [
        "easting",
        "northing",
        "depth",
        "length_x",
        "length_y",
        "length_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "mu",
        "lamda",
        "cavity_pressure",
    ]
    ro_parameters = ["volume"]

    class EllipsoidalSourceDialog(SourceEditDialog):
        def __init__(self, delegate, *args, **kwargs):
            SourceEditDialog.__init__(
                self, delegate, ui_file="ellipsoid_source.ui", *args, **kwargs
            )

            def setLabel(method, fmt, value, suffix=""):
                method(fmt.format(value) + suffix)

            self.volume.setValue = lambda v: setLabel(
                self.volume.setText, "{:.2e}", v, " m<sup>3</sup>"
            )

            self.getSourceParameters()

    ROIWidget = PointSourceROI
    EditDialog = EllipsoidalSourceDialog

    @staticmethod
    def getRepresentedSource(sandbox):
        return EllipsoidSource()

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
    <td>Dimensions</td><td>
        x: {source.length_x:.0f} m;
        y: {source.length_y:.0f} m;
        z: {source.length_z:.0f} m
    </td>
</tr><tr>
    <td>Rotations</td><td>
        x: {source.rotation_x:.1f}&deg;;
        y: {source.rotation_y:.1f}&deg;;
        z: {source.rotation_z:.1f}&deg;
    </td>
</tr><tr style="font-weight: bold;">
    <td>Volume:</td><td>{source.volume:.2e} m<sup>3</sup></td>
</tr></table>
"""
        return item.format(idx=self.index.row() + 1, delegate=self, source=self.source)


class PointCompoundSourceDelegate(SourceDelegate):

    __represents__ = "PointCompoundSource"

    display_backend = "Compound Model"
    display_name = "PointCompoundSource"

    parameters = [
        "easting",
        "northing",
        "depth",
        "dVx",
        "dVy",
        "dVz",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "nu",
    ]
    ro_parameters = ["volume"]

    class CompoundSourceDialog(SourceEditDialog):
        def __init__(self, delegate, *args, **kwargs):
            SourceEditDialog.__init__(
                self, delegate, ui_file="pCDM_source.ui", *args, **kwargs
            )

            def setLabel(method, fmt, value, suffix=""):
                method(fmt.format(value) + suffix)

            self.volume.setValue = lambda v: setLabel(
                self.volume.setText, "{:.2e}", v, " m<sup>3</sup>"
            )

            self.getSourceParameters()

    ROIWidget = PointSourceROI
    EditDialog = CompoundSourceDialog

    @staticmethod
    def getRepresentedSource(sandbox):
        return PointCompoundSource()

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
    <td>Volume Change</td><td>
        x: {source.dVx:.0f} m<sup>3</sup>;
        y: {source.dVy:.0f} m<sup>3</sup>;
        z: {source.dVz:.0f} m<sup>3</sup>
    </td>
</tr><tr>
    <td>Rotations</td><td>
        x: {source.rotation_x:.1f}&deg;;
        y: {source.rotation_y:.1f}&deg;;
        z: {source.rotation_z:.1f}&deg;
    </td>
</tr><tr style="font-weight: bold;">
    <td>Volume:</td><td>{source.volume:.2e} m<sup>3</sup></td>
</tr></table>
"""
        return item.format(idx=self.index.row() + 1, delegate=self, source=self.source)
