import numpy as num
from pyrocko.guts import Bool

from .plugin import Plugin, PluginConfig


class DerampConfig(PluginConfig):

    demean = Bool.T(optional=True, default=True)


class Deramp(Plugin):
    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or DerampConfig()

        self._log = scene._log.getChild("Deramp")

    def get_ramp_coefficients(self, displacement):
        """Fit plane through the displacement data.

        :returns: Mean of the displacement and slopes in easting coefficients
            of the fitted plane. The array hold
            ``[offset_e, offset_n, slope_e, slope_n]``.
        :rtype: :class:`numpy.ndarray`
        """
        scene = self.scene
        msk = num.isfinite(displacement)
        displacement = displacement[msk]

        coords = scene.frame.coordinates[msk.flatten()]

        # Add ones for the offset
        coords = num.hstack((num.ones_like(coords), coords))

        coeffs, res, _, _ = num.linalg.lstsq(coords, displacement, rcond=None)

        return coeffs

    def set_demean(self, demean):
        assert isinstance(demean, bool)
        self.config.demean = demean
        self.update()

    def apply(self, displacement):
        """Fit a plane onto the displacement data and subtract it

        :param demean: Demean the displacement
        :type demean: bool
        :param inplace: Replace data of the scene (default: True)
        :type inplace: bool

        :return: ``None`` if ``inplace=True`` else a new Scene
        :rtype: ``None`` or :class:`~kite.Scene`
        """
        self._log.debug("De-ramping scene")
        coeffs = self.get_ramp_coefficients(displacement)
        coords = self.scene.frame.coordinates

        ramp = coeffs[2:] * coords
        if self.config.demean:
            ramp += coeffs[:2]

        ramp = ramp.sum(axis=1).reshape(displacement.shape)

        displacement -= ramp
        return displacement
