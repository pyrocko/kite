#!/bin/python
import numpy as np
from pyrocko.guts import Float, Int, Object
from scipy import ndimage

from kite.meta import ADict, Subject


class SceneProcess(Object):
    def __init__(self, scene, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self.scene = scene
        self.evProcessChanged = Subject()

    def __setattr__(self, attr, value):
        if attr in self.T.propnames:
            self.evProcessChanged.notify()
        Object.__setattr__(self, attr, value)

    def apply(self):
        raise NotImplementedError()

    def remove(self):
        raise NotImplementedError()

    def __del__(self):
        self.evProcessChanged.unsubscribeAll()


class StaticOffset(SceneProcess):
    offset = Float.T(default=0.0, help="Static offset")

    def apply(self):
        self.scene.displacement += self.offset

    def remove(self):
        self.scene.displacement -= self.offset


class Downsample(SceneProcess):
    from scipy import ndimage

    factor = Int.T(help="Downsample factor.")

    def __init__(self, *args, **kwargs):
        SceneProcess.__init__(self, *args, **kwargs)
        sc = self.scene
        self.original = ADict()
        self.original.update(
            {
                "displacement": sc.displacement.copy(),
                "theta": sc.theta.copy(),
                "phi": sc.phi.copy(),
                "frame.dLat": sc.frame.dLat,
                "frame.dLon": sc.frame.dLon,
            }
        )

    def apply(self):
        sc = self.scene
        org = self.original
        factor = self.factor

        sx, sy = sc.displacement.shape
        gx, gy = np.ogrid[0:sx, 0:sy]
        regions = sy / factor * (gx / factor) + gy / factor
        indices = np.arange(regions.max() + 1)

        def block_downsample(arr):
            res = ndimage.mean(arr, labels=regions, index=indices)
            res.shape = (sx / factor, sy / factor)
            return res

        sc.displacement = block_downsample(sc.displacement)
        sc.theta = block_downsample(sc.theta)
        sc.phi = block_downsample(sc.phi)
        sc.frame.dLat = org["frame.dLat"] * self.factor
        sc.frame.dLon = org["frame.dLat"] * self.factor

    def remove(self):
        for prop, value in self.original.items():
            self.scene.__setattr__(prop, value)
        del self.original
