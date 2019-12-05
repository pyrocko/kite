from datetime import datetime
from .scene import BaseScene, Scene
import numpy as num


def dtime(timestamp):
    return datetime.fromtimestamp(timestamp)


class TSScene(Scene):
    pass


class SceneStack(BaseScene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._times = []
        self._active_timestamp = None
        self._scenes = {}

    @property
    def times(self):
        return sorted(self._scenes.keys())

    @property
    def tmin(self):
        return min(self.times)

    @property
    def tmax(self):
        return max(self.times)

    @property
    def timespan(self):
        return self.tmax - self.tmin

    @property
    def nscenes(self):
        return len(self._scenes)

    def add_scene(self, scene, timestamp):
        assert isinstance(scene, TSScene)
        if scene in self._scenes.values():
            raise AttributeError('Scene already in stack')

        self._log.info('Adding frame to stack at %s', dtime(timestamp))
        self._scenes[timestamp] = scene

        if self.nscenes == 1:
            self.set_scene(timestamp)

    @property
    def displacement(self):
        return self._scenes[self._active_timestamp].displacement

    @displacement.setter
    def displacement(self):
        raise AttributeError('use add_frame to set the displacement')

    def set_scene(self, timestamp):
        if timestamp not in self._scenes.keys():
            raise AttributeError('Timestamp %g not in stack' % timestamp)

        self._log.info('Setting timestamp to %s', dtime(timestamp))
        if self._active_timestamp == timestamp:
            return

        self._active_timestamp = timestamp
        self.evChanged.notify()

    def set_scene_to(self, timestamp):
        self._log.debug('Setting time to %s', dtime(timestamp))

        times_diff = num.abs(num.array(self.times) - timestamp)
        closest_timestamp = self.times[num.argmin(times_diff)]
        self.set_scene(closest_timestamp)
