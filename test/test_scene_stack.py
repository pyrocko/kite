import numpy as num
import logging
import time
from kite.scene import TestScene
from kite.scene_stack import TSScene, SceneStack

logging.basicConfig(level=logging.DEBUG)

t0 = time.time()
dt = 60*60*24*365.25 / 2


class TSTestScene(TestScene, TSScene):

    @classmethod
    def createSine(cls, *args, **kwargs):
        return super().createSine(*args, **kwargs)


nscenes = 10
scenes = {t0 - its*dt: TSTestScene.createSine()
          for its in range(nscenes)}


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
