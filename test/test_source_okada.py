import shutil
import tempfile
import unittest

import numpy as np

from kite import SandboxScene, TestSandboxScene
from kite.sources import OkadaPath, OkadaSource

from . import common

plot = False
common.setLogLevel("DEBUG")


class testSourceOkada(unittest.TestCase):
    __name__ = "SandboxTestOkada"

    def setUp(self):
        self.ms = SandboxScene()
        self.tmpdir = tempfile.mkdtemp(prefix="kite")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_okada_source(self):
        nsources = 2

        def random_int(low, high):
            return np.random.randint(low, high, size=1).astype(float)

        for s in range(nsources):
            length = random_int(5000, 15000)
            self.ms.addSource(
                OkadaSource(
                    easting=random_int(
                        self.ms.frame.E.min(), self.ms.frame.E.max()
                    ),  # ok
                    northing=random_int(
                        self.ms.frame.N.min(), self.ms.frame.N.max()
                    ),  # ok
                    depth=random_int(0, 8000),  # ok
                    strike=random_int(0, 360),  # ok
                    dip=random_int(0, 170),
                    slip=random_int(1, 5),  # ok
                    rake=random_int(0, 180),
                    length=length,
                    width=15.0 * length**0.66,
                )
            )

            self._plot_displacement(self.ms)

    def _test_okada_path(self):
        ok_path = OkadaPath(
            easting=10000,
            northing=24000,
        )
        ok_path.addNode(15000, 28000)
        ok_path.addNode(18000, 32000)
        ok_path.addNode(22000, 34000)
        # ok_path.insertNode(1, 22000, 34000)
        self.ms.addSource(ok_path)

        self._plot_displacement(self.ms)

    @staticmethod
    def _plot_displacement(ms):
        if not plot:
            ms.down
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        fig = plt.figure()
        ax = fig.gca()
        ms.processSources()

        ax.imshow(
            np.flipud(ms.down),
            aspect="equal",
            extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()],
        )
        for src in ms.sources:
            for seg in src.segments:
                p = Polygon(seg.outline(), alpha=0.8, fill=False)
                ax.add_artist(p)
            if isinstance(src, OkadaPath):
                nodes = np.array(src.nodes)
                ax.scatter(nodes[:, 0], nodes[:, 1], color="r")
        plt.show()
        fig.clear()

    def testModelSaveLoad(self):
        filename = self.tmpdir + "/testsave.yml"
        msc = TestSandboxScene.randomOkada(nsources=2)
        msc.save(filename=filename)

        msd2 = SandboxScene.load(filename=filename)  # noqa
        # print msc2.config


if __name__ == "__main__":
    plot = True
    unittest.main(exit=False)
