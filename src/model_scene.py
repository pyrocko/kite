from pyrocko.guts import Object, List
from meta import Subject
from scene import BaseScene, FrameConfig


class ModelSceneConfig(Object):
    frame = FrameConfig.T(
        default=FrameConfig(),
        help='Frame/reference configuration')
    sources = List.T(
        optional=True,
        help='List of sources')


class ModelScene(BaseScene):

    evChanged = Subject()

    def __init__(self, config=ModelSceneConfig(), **kwargs):
        self.config = config

        frame_config = kwargs.pop('frame_config', None)
        if frame_config is not None:
            self.config.frame = frame_config

        BaseScene.__init__(self, frame_config=self.config.frame, **kwargs)

    @property
    def displacement(self):
        return

    def setReferenceScene(self, scene):
        self.reference_scene = scene

    def process(self, frame, okada_source, nthreads=0):
        disloc_source = okada_source.disloc_source()

        self.log.info('Calculating Okada displacement for %d planes'
                      % disloc_source.shape[0])

    def getScene(self):
        '''Return a full featured :class:`Scene` from current model.

        :returns: Scene
        :rtype: :class:`Scene`
        '''
        return
