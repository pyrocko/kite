from datetime import datetime

import numpy as np

from .scene import Scene


def dtime(timestamp):
    return datetime.fromtimestamp(timestamp)


class TSScene(Scene):
    pass


class SceneStack(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._times = []
        self._scenes = {}

        self._selected_tmin = None
        self._selected_tmax = None

        self._range_tmin = None
        self._range_tmax = None

        self._scene_tmin = None
        self._scene_tmax = None

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
            raise AttributeError("Scene already in stack")

        self._log.info("Adding frame to stack at %s", dtime(timestamp))
        self._scenes[timestamp] = scene

        if self.nscenes == 1:
            self.phi = scene.phi
            self.theta = scene.theta
            self.rows, self.cols = scene.displacement.shape
            self.frame = scene.frame

        self.set_time_range(self.tmin, self.tmax)

    @property
    def displacement(self):
        displacement = self._scene_tmax.displacement - self._scene_tmin.displacement
        return displacement

    @property
    def displacement_mask(self):
        return ~np.isfinite(self.displacement)

    @displacement.setter
    def displacement(self):
        raise AttributeError("use add_frame to set the displacement")

    def get_scene_at(self, timestamp):
        times = self.times
        time = times[np.abs(np.array(times) - timestamp).argmin()]
        return time, self._scenes[time]

    def set_time_range(self, tmin, tmax):
        assert tmin <= tmax, "required tmin <= tmax"
        assert tmin >= self.tmin, "tmin outside of stack time range"
        assert tmax <= self.tmax, "tmax outside of stack time range"

        if self._range_tmin == tmin and self._range_tmax == tmax:
            self._log.debug("Time range unchanged")
            return

        self._log.info("Setting time range to %s - %s", dtime(tmin), dtime(tmax))

        self._selected_tmin = tmin
        self._selected_tmax = tmax

        self._range_tmin, scene_tmin = self.get_scene_at(tmin)
        self._range_tmax, scene_tmax = self.get_scene_at(tmax)

        if scene_tmin is self._scene_tmin and scene_tmax is self._scene_tmax:
            return

        self._scene_tmin = scene_tmin
        self._scene_tmax = scene_tmax

        self.evChanged.notify()
