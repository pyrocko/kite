import numpy as np
from pyrocko.guts import Tuple
from scipy import stats

from .plugin import Plugin, PluginConfig


class APSConfig(PluginConfig):
    patch_coords = Tuple.T(optional=True)
    model_coefficients = Tuple.T(optional=True)


class APS(Plugin):
    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or APSConfig()
        self._log = scene._log.getChild("APS")

    def get_elevation(self):
        return self.scene.get_elevation()

    def set_patch_coords(self, xmin, ymin, xsize, ysize):
        self.config.patch_coords = (xmin, ymin, xsize, ysize)

    def get_patch_coords(self):
        if self.config.patch_coords is None:
            frame = self.scene.frame
            scene = self.scene
            rstate = np.random.RandomState(123)

            while True:
                llE = rstate.uniform(0, frame.lengthE * (4 / 5))
                llN = rstate.uniform(0, frame.lengthN * (4 / 5))
                urE = frame.lengthE / 5
                urN = frame.lengthN / 5

                colmin, colmax = frame.mapENMatrix(llE, llE + urE)
                rowmin, rowmax = frame.mapENMatrix(llN, llN + urN)

                displacement = scene._displacement[rowmin:rowmax, colmin:colmax]
                if np.any(displacement):
                    return llE, llN, urE, urN

        return self.config.patch_coords

    def get_data(self):
        scene = self.scene
        frame = self.scene.frame
        coords = self.get_patch_coords()
        if not coords:
            raise AttributeError("Set coordinates for APS.")

        colmin, colmax = frame.mapENMatrix(coords[0], coords[0] + coords[2])
        rowmin, rowmax = frame.mapENMatrix(coords[1], coords[1] + coords[3])
        displacement = scene._displacement[rowmin:rowmax, colmin:colmax]
        elevation = self.get_elevation()[rowmin:rowmax, colmin:colmax]

        mask = np.isfinite(displacement)
        elevation = elevation[mask]
        displacement = displacement[mask]

        return elevation, displacement

    def get_correlation(self):
        elevation, displacement = self.get_data()
        slope, intercept, _, _, _ = stats.linregress(
            elevation.ravel(), displacement.ravel()
        )

        return slope, intercept

    def apply(self, displacement):
        self._log.info("Applying APS model to displacement")

        scene = self.scene
        elevation = scene.get_elevation()
        slope, intercept = self.get_correlation()

        correction = elevation * slope + intercept
        displacement -= correction
        return displacement
