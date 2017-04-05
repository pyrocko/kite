import numpy as num
import time

from pyrocko.guts import Object, List, Int
from pyrocko import guts
from meta import Subject, property_cached
from scene import BaseScene, FrameConfig
from os import path as op

# Import the modeling backends
from .sources import DislocProcessor


__processors__ = [DislocProcessor]

km = 1e3


class ModelSceneConfig(Object):
    frame = FrameConfig.T(
        default=FrameConfig(),
        help='Frame/reference configuration')
    extent_north = Int.T(
        default=800,
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
        self.evModelUpdated = Subject()
        BaseScene.__init__(self, frame_config=self.config.frame, **kwargs)

        self.reference = None
        self._los_factors = None

        for attr in ['theta', 'phi']:
            data = kwargs.pop(attr, None)
            if data is not None:
                self.__setattr__(attr, data)

        self.setExtent(self.config.extent_north, self.config.extent_east)

    @property
    def sources(self):
        return self.config.sources

    def setExtent(self, east, north):
        if self.reference is not None:
            self._log.warning('Cannot change a referenced model!')
            return
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
        self._resetLOSFactors()

        self.frame._updateExtent()
        self._clearModel()
        self.evChanged.notify()

    @property_cached
    def los_displacement(self):
        self.processSources()
        los_factors = self._LOSFactors()

        self._los_displacement =\
            (los_factors[:, :, 0] * -self.down +
             los_factors[:, :, 1] * self.east +
             los_factors[:, :, 2] * self.north)
        return self._los_displacement

    def addSource(self, source):
        if source not in self.sources:
            self.sources.append(source)
        self._log.info('Added %s' % source.__class__.__name__)
        source.evParametersChanged.subscribe(self._clearModel)

        self._clearModel()

    def removeSource(self, source):
        source.evParametersChanged.unsubscribe(self._clearModel)
        self.sources.remove(source)
        self._log.info('Removed %s' % source.__class__.__name__)
        del source

        self._clearModel()

    def processSources(self):
        results = []
        for processor in __processors__:
            sources = [src for src in self.sources
                       if src.__implements__ == processor.__implements__]
            if not sources:
                continue

            t0 = time.time()

            result = processor.process(
                sources, self.frame.coordinates, nthreads=0)
            results.append(result)

            self._log.debug('Processed %s (nsources:%d) using %s [%.4f s]'
                            % (src.__class__.__name__, len(sources),
                               processor.__name__, time.time() - t0))

        for r in results:
            self.north += r['north'].reshape(self.rows, self.cols)
            self.east += r['east'].reshape(self.rows, self.cols)
            self.down += r['down'].reshape(self.rows, self.cols)

    def loadReferenceScene(self, filename):
        from .scene import Scene
        scene = Scene.load(filename)
        self.setReferenceScene(scene)

    def setReferenceScene(self, scene):
        self.setExtent(scene.cols, scene.rows)

        self.frame._updateConfig(scene.frame.config)

        self.phi = scene.phi
        self.theta = scene.theta
        self._resetLOSFactors()
        self.reference = Reference(self, scene)

        self._clearModel()

    def getKiteScene(self):
        '''Return a full featured :class:`Scene` from current model.

        :returns: Scene
        :rtype: :class:`Scene`
        '''
        from .scene import Scene, SceneConfig
        self._log.info('Creating kite.Scene from ModelScene')

        config = SceneConfig()
        config.frame = self.frame.config
        config.meta.scene_id = 'Exported ModelScene'

        return Scene(
            displacement=self.los_displacement,
            theta=self.theta,
            phi=self.phi,
            config=config)

    def _clearModel(self):
        for arr in [self.north, self.east, self.down]:
            arr.fill(0.)
        self.los_displacement = None
        self.evModelUpdated.notify()

    def _resetLOSFactors(self):
        self._los_factors = None

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

    def save(self, filename):
        _file, ext = op.splitext(filename)
        filename = filename if ext in ['.yml'] else filename + '.yml'
        self._log.info('Saving model scene to %s' % filename)
        self.config.dump(filename='%s' % filename,
                         header='kite.ModelScene YAML Config')

    @classmethod
    def load(cls, filename):
        config = guts.load(filename=filename)
        model_scene = cls(config=config)
        for source in model_scene.sources:
            model_scene.addSource(source)
        model_scene._log.info('Loaded config from %s' % filename)
        return model_scene


class Reference(object):
    def __init__(self, model, scene):
        self.model = model
        self.scene = scene

        self.model.evModelUpdated.subscribe(self._clearRefence)

    def _clearRefence(self):
        self.difference = None

    @property_cached
    def difference(self):
        return self.scene.displacement - self.model.los_displacement


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
                    dip=r(0, 90),
                    slip=r(1, 5),
                    rake=r(-180, 180),
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
