#!/bin/python


class SceneProcess(object):
    def __init__(self, scene):
        self._scene = scene
        self.parameters = {}

    def process(self):
        pass


class StaticCorrection(SceneProcess):
    def __init__(self, scene):
        SceneProcess.__init__(self, scene)
        self.parameters['static_correction'] = 0

    def process(self):
        self.scene.displacement += self.parameters['static_correction']
