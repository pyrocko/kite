import matplotlib.pyplot as plt
import numpy as num
from matplotlib.path import Path
from pyrocko.guts import Bool, Dict

from .plugin import Plugin, PluginConfig


class PolygonMaskConfig(PluginConfig):
    polygons = Dict.T(optional=True, default={})
    applied = Bool.T(default=True)


class PolygonMask(Plugin):
    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or PolygonMaskConfig()
        self._points = None

        self._log = scene._log.getChild("Mask")

    @property
    def polygons(self):
        return self.config.polygons

    @property
    def npolygons(self):
        return len(self.config.polygons)

    def get_points(self):
        if self._points is None:
            f = self.scene.frame
            cols, rows = num.meshgrid(range(f.cols), range(f.rows))
            self._points = num.vstack((cols.ravel(), rows.ravel())).T
        return self._points

    def click_one_polygon(self):
        """
        Open a matplotlib window to click a closed polygon to mask.
        """

        sc = self.scene

        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.imshow(sc.displacement, origin="lower")
        ax.set_title("Click to add vertex. Press ENTER to finish.")

        #  Click polygon to mask
        vertices = plt.ginput(-1)
        self.add_polygon(vertices)

    def add_polygon(self, vertices):
        pid = len(self.config.polygons)
        while pid in self.config.polygons:
            pid += 1
        self.config.polygons[pid] = vertices
        self.update()

        return pid

    def remove_polygon(self, pid):
        self.config.polygons.pop(pid, None)
        self.update()

    def update_polygon(self, pid, vertices):
        if pid not in self.config.polygons:
            return
        self.config.polygons[pid] = vertices
        self.update()

    def apply(self, displacement):
        sc = self.scene
        points = self.get_points()

        mask = num.full((sc.frame.rows, sc.frame.cols), False)
        for vertices in self.config.polygons.values():
            p = Path(vertices)
            mask |= p.contains_points(points).reshape(sc.frame.rows, sc.frame.cols)

        displacement[mask] = num.nan
        return displacement
