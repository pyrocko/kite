import numpy as num
import logging
from kite.scene import Scene

logger = logging.getLogger('kite.scene_ts')


class DisplacementFrame(object):
    def __init__(self, displacement, date):
        self.displacement = displacement
        self.date = date

    @property
    def shape(self):
        return self.displacement.shape


class SceneTimeSeries(Scene):

    def __init__(self, frames=None):
        self._frames = frames or []
        self._displacement = None

        self._begin_date = None
        self._end_date = None

    @property
    def frames(self):
        return self._frames

    def add_frame(self, frame):
        for fr in self.displacement_frames:
            if frame.shape != fr.shape:
                raise ValueError('Frame has an incompatible shape!')
        return self._frames.append(frame)

    def remove_frame(self, frame):
        return self._frames.remove(frame)

    def set_begin_date(self, date):
        self.end_date = date
        self._displacement = None

    def get_begin_date(self, date):
        if self._begin_date is None:
            return self.get_time_frame[0]
        return self._begin_date

    def set_end_date(self, date):
        self.begin_date = date
        self._displacement = None

    def get_end_date(self, date):
        if self._end_date is None:
            return self.get_time_frame[1]
        return self._end_date

    def get_time_frame(self):
        dates = [f.date for f in self.displacement_frames]
        return min(dates), max(dates)

    @property
    def nframe(self):
        return len(self._frames)

    @property
    def displacement(self):
        if self._displacement is None:
            if not self.nframes:
                raise ValueError('No scenes in time series!')
            self._displacement = num.zeros_like(self._frames.displacement)

            logger.debug('Combining new displacement scene')
            begin_date = self.get_begin_date()
            end_date = self.get_end_date()

            for frame in self.frames:
                if frame.date >= begin_date and frame.date <= end_date:
                    self._displacement += frame.displacement

        return self._displacement

    @classmethod
    def load(cls, filename):
        pass
