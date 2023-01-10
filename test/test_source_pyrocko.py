import shutil
import tempfile
import unittest

import numpy as np

from kite import SandboxScene, TestSandboxScene  # noqa
from kite.sources import (
    PyrockoDoubleCouple,
    PyrockoMomentTensor,
    PyrockoRectangularSource,
    PyrockoRingfaultSource,
)

from . import common

plot = False
gf_store = "/home/marius/Development/testing/leeds/aquila_example/insar/gf_abruzzo_nearfield"  # noqa

common.setLogLevel("DEBUG")


class testSourcePyrocko(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ms = SandboxScene()
        cls.tmpdir = tempfile.mkdtemp(prefix="kite")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_rectangular_source(self):
        nsources = 2

        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        for s in range(nsources):
            length = r(5000, 15000)
            self.ms.addSource(
                PyrockoRectangularSource(
                    easting=r(0.0, self.ms.frame.E.max()),  # ok
                    northing=r(0.0, self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    slip=r(1, 7),  # ok
                    rake=r(0, 180),
                    length=length,
                    width=15.0 * length**0.66,
                    store_dir=gf_store,
                )
            )  # noqa

        self._plot_displacement(self.ms)

    def test_moment_tensor(self):
        nsources = 5

        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        for s in range(nsources):
            self.ms.addSource(
                PyrockoMomentTensor(
                    easting=r(0.0, self.ms.frame.E.max()),  # ok
                    northing=r(0.0, self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    store_dir=gf_store,
                )
            )

        self._plot_displacement(self.ms)

    def test_double_couple(self):
        nsources = 5

        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        for s in range(nsources):
            self.ms.addSource(
                PyrockoDoubleCouple(
                    easting=r(0.0, self.ms.frame.E.max()),  # ok
                    northing=r(0.0, self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    store_dir=gf_store,
                )
            )

        self._plot_displacement(self.ms)

    def test_ring_fault(self):
        nsources = 1

        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        for s in range(nsources):
            diameter = r(5000, 15000)
            self.ms.addSource(
                PyrockoRingfaultSource(
                    easting=r(0.0, self.ms.frame.E.max()),  # ok
                    northing=r(0.0, self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    magnitude=r(2, 6),  # ok
                    diameter=diameter,
                    store_dir=gf_store,
                )
            )  # noqa

        self._plot_displacement(self.ms)

    @staticmethod
    def _plot_displacement(ms):
        if not plot:
            ms.down
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon  # noqa

        fig = plt.figure()
        ax = fig.gca()
        ms.processSources()

        ax.imshow(
            np.flipud(ms.north),
            aspect="equal",
            extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()],
        )
        # for src in ms.sources:
        #     for seg in src.segments:
        #         p = Polygon(seg.outline(), alpha=.8, fill=False)
        #         ax.add_artist(p)
        plt.show()
        fig.clear()


if __name__ == "__main__":
    plot = True
    unittest.main(exit=False)
