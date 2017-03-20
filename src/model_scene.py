from pyrocko.guts import Object


class ModelScene(Object):

    @property
    def residual(self):
        return self.ref_scene.displacement - self.displacement

    def process(self, frame, okada_source, nthreads=0):
        disloc_source = okada_source.disloc_source()

        self.log.info('Calculating Okada displacement for %d planes'
                      % disloc_source.shape[0])

    def getScene(self):
        '''Return a full featured :class:`Scene` from model.
        
        :returns: Scene
        :rtype: :class:`Scene`
        '''
