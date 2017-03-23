import numpy as num

from pyrocko.guts import Object, List, Int
from meta import Subject, property_cached
from scene import BaseScene, FrameConfig

# Import the modeling backends
from .sources import DislocProcessor


processors_available = [DislocProcessor]
km = 1e3


class ModelSceneConfig(Object):
    frame = FrameConfig.T(
        default=FrameConfig(),
        help='Frame/reference configuration')
    extent_north = Int.T(
        default=900,
        help='Model size towards north in [px]')
    extent_east = Int.T(
        default=800,
        help='Model size towards east in [px]')
    sources = List.T(
        help='List of sources')


class ModelScene(BaseScene):

    evChanged = Subject()
    evConfigChanged = Subject()

    def __init__(self, config=ModelSceneConfig(), **kwargs):
        self.config = config
        self.sources = self.config.sources
        self.evModelChanged = Subject()

        frame_config = kwargs.pop('frame_config', None)
        if frame_config is not None:
            self.config.frame = frame_config
        BaseScene.__init__(self, frame_config=self.config.frame, **kwargs)

        self.setExtent(self.config.extent_north, self.config.extent_east)

        for attr in ['theta', 'phi']:
            data = kwargs.pop(attr, None)
            if data is not None:
                self.__setattr__(attr, data)

        self._los_factors = None

    def setExtent(self, north, east):
        self.config.extent_east = east
        self.config.extent_north = north

        self.cols = east
        self.rows = north

        self.north = num.zeros((self.rows, self.cols))
        self.east = num.zeros_like(self.north)
        self.down = num.zeros_like(self.north)

        self._theta = num.zeros_like(self.north)
        self._phi = num.zeros_like(self.north)
        self._theta.fill(num.pi/2)
        self._phi.fill(0.)
        self._los_factors = None

        self.evChanged.notify()

    @property_cached
    def displacement(self):
        self.processGrid()
        los_fac = self._LOSFactors()

        self._displacement =\
            (los_fac[:, :, 0] * -self.down +
             los_fac[:, :, 1] * self.east +
             los_fac[:, :, 2] * self.north)
        return self._displacement

    def addSource(self, source):
        self.sources.append(source)
        source.evParametersChanged.subscribe(self._clearModel)

    def processGrid(self):
        results = []
        for processor in processors_available:
            sources = [s for s in self.sources
                       if s.__implements__ == processor.__implements__]
            result = processor.process(
                sources, self.frame.coordinates, nthreads=0)
            results.append(result)

        for r in results:
            self.north += r['north'].reshape(self.rows, self.cols)
            self.east += r['east'].reshape(self.rows, self.cols)
            self.down += r['down'].reshape(self.rows, self.cols)

    def getScene(self):
        '''Return a full featured :class:`Scene` from current model.

        :returns: Scene
        :rtype: :class:`Scene`
        '''
        from .scene import Scene, SceneConfig

        config = SceneConfig()
        config.frame = self.frame
        config.meta.scene_id = 'Model Scene'

        return Scene(
            displacement=self.displacement,
            theta=self.theta,
            phi=self.phi,
            config=config)

    def _clearModel(self):
        for arr in [self.north, self.east, self.down]:
            arr.fill(0.)
        self.displacement = None
        self.evModelChanged.notify()

    def _LOSFactors(self):
        if (self.theta.size != self.phi.size):
            raise AttributeError('LOS angles inconsistent with provided'
                                 ' coordinate shape.')
        if self._los_factors is None:
            self._los_factors = num.empty((self.theta.shape[0],
                                           self.theta.shape[1],
                                           3))
            self._los_factors[:, :, 0] = num.sin(self.theta)
            self._los_factors[:, :, 1] = num.cos(self.theta)\
                * num.cos(self.phi)
            self._los_factors[:, :, 2] = num.cos(self.theta)\
                * num.sin(self.phi)
        return self._los_factors


class TestModelScene(ModelScene):

    @classmethod
    def randomOkada(cls, nsources=1):
        from .sources import OkadaSource
        model_scene = cls()

        def r(lo, hi):
            return float(num.random.randint(
                lo, high=hi, size=1))

        for s in xrange(nsources):
            length = r(5000, 15000)
            model_scene.addSource(
                OkadaSource(
                    easting=r(length, model_scene.frame.E.max()-length),
                    northing=r(length, model_scene.frame.N.max()-length),
                    depth=r(0, 8000),
                    strike=r(0, 360),
                    dip=r(0, 170),
                    slip=r(1, 5),
                    rake=r(0, 180),
                    length=length,
                    width=15. * length**.66,))
        return model_scene

    @classmethod
    def simpleOkada(cls, **kwargs):
        from .sources import OkadaSource
        model_scene = cls()

        parameters = {
            'easting': 50000,
            'northing': 50000,
            'depth': 0,
            'strike': 180.,
            'dip': 20,
            'slip': 2,
            'rake': 90,
            'length': 10000,
            'width': 15. * 10000**.66,
        }
        parameters.update(kwargs)

        model_scene.addSource(OkadaSource(**parameters))
        return model_scene


class ProcessorProfile(dict):
    pass


class ModelProcessor(object):
    '''Interface definition of the processor '''
    __implements__ = 'disloc'  # Defines What backend is implemented

    @staticmethod
    def process(sources, coords, nthreads=0):
        raise NotImplementedError()

        result = {
            'north': num.array(),
            'east': num.array(),
            'down': num.array(),
            'processor_profile': ProcessorProfile()
        }
        return result
