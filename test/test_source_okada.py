import shutil
import tempfile
import unittest

import numpy as num

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
        print(self.tmpdir)

    def tearDown(self):
        return
        shutil.rmtree(self.tmpdir)

    def test_okada_source(self):
        nsources = 2

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            length = r(5000, 15000)
            self.ms.addSource(
                OkadaSource(
                    easting=r(0.0, self.ms.frame.E.max()),  # ok
                    northing=r(0.0, self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    slip=r(1, 5),  # ok
                    rake=r(0, 180),
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
            num.flipud(ms.down),
            aspect="equal",
            extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()],
        )
        for src in ms.sources:
            for seg in src.segments:
                p = Polygon(seg.outline(), alpha=0.8, fill=False)
                ax.add_artist(p)
            if isinstance(src, OkadaPath):
                nodes = num.array(src.nodes)
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
