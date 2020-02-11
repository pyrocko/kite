import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as num

from pyrocko.guts import Object, Dict, Bool
from kite.util import Subject


class PolygonMaskConfig(Object):
    polygons = Dict.T(optional=True, default={})
    applied = Bool.T(default=True)


class PolygonMask(object):

    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or PolygonMaskConfig()
        self._points = None 
        
        self._log = scene._log.getChild('Mask')

        self.evChanged = Subject()
    
    @property
    def npolygons(self):
        return len(self.config.polygons)

    def get_points(self):
        if self._points is None:
            f = self.scene.frame
            self._points = [(c, r) for r in range(f.rows)
                            for c in range(f.cols)]
#            rows, cols = num.meshgrid(f.rows, f.cols)
#            rows = rows.ravel()
#            cols = cols.ravel()
#
#            self._points
        return self._points

    def click_one_polygon(self):
        """
        Open a matplotlib window to click a closed polygon to mask.
        """

        sc = self.scene

        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.imshow(sc.displacement, origin='lower')
        ax.set_title('Click to add vertex. Press ENTER to finish.')

        #  Click polygon to mask
        polygon = plt.ginput(-1)
        print(polygon)
        self.add_polygon(polygon)

    def add_polygon(self, polygon):
        pid = len(self.config.polygons)
        while pid in self.config.polygons:
            pid += 1
        print(pid)

        self.config.polygons[pid] = polygon

    def remove_polygon(self, pid):
        self.config.polygons.pop(pid, None)

    def is_applied(self):
        return self.config.applied

    def get_mask(self):
        if not self.config.applied or not self.npolygons:
            return None
        sc = self.scene
        points = self.get_points()

        mask = num.full((sc.frame.rows, sc.frame.cols), False)
        for polygon in self.config.polygons.values():
            p = Path(polygon)
            mask |= p.contains_points(points).reshape(
                    sc.frame.rows, sc.frame.cols)

        return mask
