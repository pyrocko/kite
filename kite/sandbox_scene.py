import time
from os import path as op

import numpy as np
from pyrocko import guts
from pyrocko.guts import Int, List, Object, String

from kite.scene import BaseScene, FrameConfig

# Import the modeling backends
from kite.sources import CompoundModelProcessor, DislocProcessor, PyrockoProcessor
from kite.util import Subject, property_cached

__processors__ = [DislocProcessor, PyrockoProcessor, CompoundModelProcessor]
km = 1e3
d2r = np.pi / 180.0
r2d = 180.0 / np.pi


class SandboxSceneConfig(Object):
    frame = FrameConfig.T(default=FrameConfig(), help="Frame/reference configuration")
    extent_north = Int.T(default=800, help="Model size towards north in [px]")
    extent_east = Int.T(default=800, help="Model size towards east in [px]")
    sources = List.T(help="List of sources")
    reference_scene = String.T(optional=True, help="Reference kite.Scene container")


class SandboxScene(BaseScene):
    def __init__(self, config=None, **kwargs):
        self.evChanged = Subject()
        self.evModelUpdated = Subject()
        self.evConfigChanged = Subject()
        self._initialised = False

        self.config = config if config else SandboxSceneConfig()
        super().__init__(frame_config=self.config.frame, **kwargs)

        self.reference = None
        self._los_factors = None

        for attr in ("theta", "phi"):
            data = kwargs.pop(attr, None)
            if data is not None:
                self.__setattr__(attr, data)

        self.setExtent(self.config.extent_east, self.config.extent_north)

        if self.config.reference_scene is not None:
            self.loadReferenceScene(self.config.reference_scene)

    @property
    def sources(self):
        """
        :returns: List of sources attached sandbox
        :rtype: list
        """
        return self.config.sources

    def setExtent(self, east, north):
        """Set the sandbox's extent in pixels

        :param east: Pixels in East
        :type east: int
        :param north: Pixels in North
        :type north: int
        """
        if self.reference is not None:
            self._log.warning("Cannot change a referenced model!")
            return

        self._log.debug("Changing model extent to %d px by %d px" % (east, north))

        self.cols = east
        self.rows = north

        self._north = np.zeros((self.rows, self.cols))
        self._east = np.zeros_like(self._north)
        self._down = np.zeros_like(self._north)

        self.theta = np.zeros_like(self._north)
        self.phi = np.zeros_like(self._north)
        self.theta.fill(np.pi / 2)
        self.phi.fill(0.0)

        self.config.extent_east = east
        self.config.extent_north = north

        self.frame.updateExtent()
        self._clearModel()
        self.processSources()

        self.evChanged.notify()

    def setLOS(self, phi, theta):
        """Set the sandbox's LOS vector

        :param phi: phi in degree
        :type phi: int
        :param theta: theta in degree
        :type theta: int
        """
        if self.reference is not None:
            self._log.warning("Cannot change a referenced model!")
            return

        self._log.debug("Changing model LOS to %d phi and %d theta", phi, theta)

        self.theta = np.full_like(self.theta, theta * r2d)
        self.phi = np.full_like(self.phi, phi * r2d)
        self.frame.updateExtent()

        self._clearModel()
        self.evChanged.notify()

    @property
    def north(self):
        if not self._initialised:
            self.processSources()
        return self._north

    @property
    def east(self):
        if not self._initialised:
            self.processSources()
        return self._east

    @property
    def down(self):
        if not self._initialised:
            self.processSources()
        return self._down

    @property_cached
    def displacement(self):
        """Displacement projected to LOS"""
        self.processSources()
        los_factors = self.los_rotation_factors

        self._displacement = (
            los_factors[:, :, 0] * -self._down
            + los_factors[:, :, 1] * self._east
            + los_factors[:, :, 2] * self._north
        )
        return self._displacement

    @property_cached
    def max_horizontal_displacement(self):
        """Maximum horizontal displacement"""
        return np.sqrt(self._north**2 + self._east**2).max()

    def addSource(self, source):
        """Add displacement source to sandbox

        :param source: Displacement Source
        :type source: :class:`kite.sources.meta.SandboxSource`
        """
        if source not in self.sources:
            self.sources.append(source)
        source._sandbox = self

        source.evParametersChanged.subscribe(self._clearModel)
        self._clearModel()

        self._log.debug("Source %s added" % source.__class__.__name__)

    def removeSource(self, source):
        """Remove displacement source from sandbox

        :param source: Displacement Source
        :type source: :class:`kite.sources.meta.SandboxSource`
        """
        source.evParametersChanged.unsubscribe(self._clearModel)
        self.sources.remove(source)
        self._log.debug("Source %s removed" % source.__class__.__name__)
        del source

        self._clearModel()

    def processSources(self):
        """Process displacement sources and update displacements"""
        result = self._process(self.sources)

        self._north += result["north"].reshape(self.rows, self.cols)
        self._east += result["east"].reshape(self.rows, self.cols)
        self._down += result["down"].reshape(self.rows, self.cols)
        self._initialised = True

    def processCustom(self, coordinates, sources, result_dict=None):
        return self._process(sources, result_dict)

    def _process(self, sources, result=None):
        if result is None:
            result = np.zeros(
                self.frame.npixel,
                dtype=[
                    ("north", float),
                    ("east", float),
                    ("down", float),
                ],
            )

        avail_processors = {}
        for proc in __processors__:
            avail_processors[proc.__implements__] = proc

        for impl in set([src.__implements__ for src in sources]):
            proc_sources = [
                src
                for src in sources
                if src.__implements__ == impl and src._cached_result is None
            ]

            if not proc_sources:
                continue

            processor = avail_processors.get(impl, None)

            if processor is None:
                self._log.warning("Could not find source processor for %s", impl)
                continue

            t0 = time.time()

            proc_result = processor(self).process(
                proc_sources, sandbox=self, nthreads=0
            )

            src_name = proc_sources[0].__class__.__name__
            self._log.debug(
                "Processed %s (nsources:%d) using %s [%.4f s]",
                src_name,
                len(proc_sources),
                processor.__name__,
                time.time() - t0,
            )

            result["north"] += proc_result["displacement.n"]
            result["east"] += proc_result["displacement.e"]
            result["down"] += proc_result["displacement.d"]

        return result

    def loadReferenceScene(self, filename):
        """Load a reference kite scene container into the sandbox

        A reference scene could be actually measured InSAR displacements.

        :param filename: filename of the scene container to load [.npy, .yml]
        :type filename: str
        """
        from .scene import Scene

        self._log.debug("Loading reference scene from %s", filename)
        scene = Scene.load(filename)
        self.setReferenceScene(scene)
        self.config.reference_scene = filename

    def setReferenceScene(self, scene):
        """Set a reference scene.

        A reference scene could be actually measured InSAR displacements.

        :param scene: Kite scene
        :type scene: :class:`kite.Scene`
        """
        self.config.frame = scene.config.frame

        self.setExtent(scene.cols, scene.rows)
        self.frame._updateConfig()

        self.phi = scene.phi
        self.theta = scene.theta
        self._los_factors = None
        self.reference = Reference(self, scene)
        self._log.debug("Reference scene set to scene.id:%s", scene.meta.scene_id)
        self.config.reference_scene = scene.meta.filename

        self._clearModel()

    def getKiteScene(self):
        """Return a :class:`kite.Scene` from current model.

        :returns: Scene
        :rtype: :class:`Scene`
        """
        from .scene import Scene, SceneConfig

        self._log.debug("Creating kite.Scene from SandboxScene")

        config = SceneConfig()
        config.frame = self.frame.config
        config.meta.scene_id = "Exported SandboxScene"

        return Scene(
            displacement=self.displacement,
            theta=self.theta,
            phi=self.phi,
            config=config,
        )

    def _clearModel(self):
        for arr in (self._north, self._east, self._down):
            arr.fill(0.0)
            if self.reference:
                arr[self.reference.scene.displacement_mask] = np.nan

        self.displacement = None
        self._los_factors = None
        self._initialised = False

        self.max_horizontal_displacement = None

        self.evModelUpdated.notify()

    def save(self, filename):
        """Save the sandbox as kite scene container

        :param filename: filename to save under
        :type filename: str
        """
        _file, ext = op.splitext(filename)
        filename = filename if ext in [".yml"] else filename + ".yml"
        self._log.debug("Saving model scene to %s" % filename)
        for source in self.sources:
            source.regularize()
        self.config.dump(
            filename="%s" % filename, header="kite.SandboxScene YAML Config"
        )

    @classmethod
    def load(cls, filename):
        """Load a :class:`kite.SandboxScene`

        :param filename: Config file to load [.yml]
        :type filename: str
        :returns: A sandbox from config file
        :rtype: :class:`kite.SandboxScene`
        """
        config = guts.load(filename=filename)
        sandbox_scene = cls(config=config)
        sandbox_scene._log.debug("Loading config from %s" % filename)
        for source in sandbox_scene.sources:
            sandbox_scene.addSource(source)
        return sandbox_scene


class Reference(object):
    def __init__(self, model, scene):
        self.model = model
        self.scene = scene

        self.model.evModelUpdated.subscribe(self._clearRefence)

    def _clearRefence(self):
        self.difference = None

    @property_cached
    def difference(self):
        return self.scene.displacement - self.model.displacement

    def optimizeSource(self, callback=None):
        from scipy import optimize

        quadtree = self.scene.quadtree
        coordinates = quadtree.leaf_coordinates
        sources = self.model.sources

        model_result = np.zeros(
            coordinates.shape[0],
            dtype=[
                ("north", float),
                ("east", float),
                ("down", float),
            ],
        )

        if len(sources) > 1 or not sources:
            self._log.warning("We can optimize single, individual sources only!")
            return
        source = sources[0]
        source.evParametersChanged.mute()

        def misfit(model_displacement, lp_norm=2):
            p = lp_norm
            mf = (
                np.sum(np.abs(quadtree.leaf_medians - model_displacement) ** p) ** 1.0
                / p
            )
            return mf

        def kernel(model):
            source.setParametersArray(model)
            res = self.model.processCustom(coordinates, [source], model_result)

            model_displacement = (
                quadtree.leaf_los_rotation_factors[:, 0] * -res["down"]
                + quadtree.leaf_los_rotation_factors[:, 1] * res["east"]
                + quadtree.leaf_los_rotation_factors[:, 2] * res["north"]
            )
            mf = misfit(model_displacement)

            model_result.fill(0.0)
            return mf

        result = optimize.minimize(
            kernel,
            source.getParametersArray(),
            method="SLSQP",
            bounds=None,
            constraints=(),
            tol=None,
            callback=callback,
            options={
                "disp": True,
                "iprint": 10,
                "eps": 1.4901161193847656e-04,
                "maxiter": 300,
                "ftol": 1e-06,
            },
        )

        source.evParametersChanged.unmute()
        self.model.sources[0].setParametersArray(result.x)
        return result


class TestSandboxScene(SandboxScene):
    @classmethod
    def randomOkada(cls, nsources=1):
        from .sources import OkadaSource

        sandbox_scene = cls()

        def r(lo, hi):
            return float(np.random.randint(lo, high=hi, size=1))

        for isrc in range(nsources):
            length = r(5000, 15000)
            sandbox_scene.addSource(
                OkadaSource(
                    easting=r(sandbox_scene.frame.E.min(), sandbox_scene.frame.E.max()),
                    northing=r(
                        sandbox_scene.frame.N.min(), sandbox_scene.frame.N.max()
                    ),
                    depth=r(0, 8000),
                    strike=r(0, 360),
                    dip=r(0, 90),
                    slip=r(1, 5),
                    rake=r(-180, 180),
                    length=length,
                    width=15.0 * length**0.66,
                )
            )
        return sandbox_scene

    @classmethod
    def simpleOkada(cls, **kwargs):
        from .sources import OkadaSource

        sandbox_scene = cls()

        parameters = {
            "easting": 50000,
            "northing": 50000,
            "depth": 0,
            "strike": 180.0,
            "dip": 20,
            "slip": 2,
            "rake": 90,
            "length": 10000,
            "width": 15.0 * 10000**0.66,
        }
        parameters.update(kwargs)

        sandbox_scene.addSource(OkadaSource(**parameters))
        return sandbox_scene
