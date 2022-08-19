import logging
import time

import numpy as num

from kite.quadtree import QuadNode
from kite.scene import TestScene
from kite.scene_stack import SceneStack, TSScene
from kite.spool import spool

logging.basicConfig(level=logging.DEBUG)
QuadNode.MIN_PIXEL_LENGTH_NODE = 32

t0 = time.time()
dt = 60 * 60 * 24 * 365.25 / 2


class TSTestScene(TestScene, TSScene):
    @classmethod
    def createSine(cls, *args, **kwargs):
        return super().createSine(*args, **kwargs)

    @classmethod
    def createRandom(cls, *args, **kwargs):
        return super().createRandom(*args, **kwargs)

    @classmethod
    def createGauss(cls, *args, **kwargs):
        return super().createGauss(*args, **kwargs)

    @classmethod
    def createFractal(cls, *args, **kwargs):
        return super().createFractal(*args, **kwargs)


nscenes = 10
scenes = {}

for isc in range(nscenes):
    t = t0 - isc * dt
    sc = TSTestScene.createRandom(256, 256)
    sc.addNoise(1.0)

    scenes[t] = sc


stack = SceneStack()
for ts, scene in scenes.items():
    stack.add_scene(scene, timestamp=ts)


def test_set_scene():
    stack.set_scene(stack.times[-1])


def test_set_scene_to():
    tmin = min(stack.times)
    tmax = max(stack.times)

    times = num.linspace(tmin, tmax, 30)
    for ts in times:
        stack.set_scene_to(ts)


spool(stack)
