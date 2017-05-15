import unittest
import numpy as num
import tempfile
import shutil

from kite import ModelScene, TestModelScene  # noqa
from kite.sources import (PyrockoRectangularSource, PyrockoMomentTensor,
                          PyrockoDoubleCouple, PyrockoRingfaultSource)

gf_store = '/home/marius/Development/testing/leeds/aquila_example/insar/gf_abruzzo_nearfield'  # noqa


class testSourcePyrocko(unittest.TestCase):
    def setUp(self):
        self.ms = ModelScene()
        self.tmpdir = tempfile.mkdtemp(prefix='kite')
        print self.tmpdir

    def tearDown(self):
        return
        shutil.rmtree(self.tmpdir)

    def testPyrockoRectangularSource(self):
        nsources = 2

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            length = r(5000, 15000)
            self.ms.addSource(
                PyrockoRectangularSource(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    slip=r(1, 7),  # ok
                    rake=r(0, 180),
                    length=length,
                    width=15. * length**.66,
                    store_dir=gf_store))  # noqa

        # self.plotDisplacement(self.ms)

    def testPyrockoMomentTensor(self):
        nsources = 5

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            self.ms.addSource(
                PyrockoMomentTensor(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    store_dir=gf_store))

        self.plotDisplacement(self.ms)

    def testPyrockoDoubleCouple(self):
        nsources = 5

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            self.ms.addSource(
                PyrockoDoubleCouple(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    store_dir=gf_store))

        self.plotDisplacement(self.ms)

    def testPyrockoRingfault(self):
        nsources = 1

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            diameter = r(5000, 15000)
            self.ms.addSource(
                PyrockoRingfaultSource(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    magnitude=r(2, 6),  # ok
                    diameter=diameter,
                    store_dir=gf_store))  # noqa

        self.plotDisplacement(self.ms)

    @staticmethod
    def plotDisplacement(ms):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        fig = plt.figure()
        ax = fig.gca()
        ms.processSources()

        ax.imshow(num.flipud(ms.north), aspect='equal',
                  extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()])
        # for src in ms.sources:
        #     for seg in src.segments:
        #         p = Polygon(seg.outline(), alpha=.8, fill=False)
        #         ax.add_artist(p)
        plt.show()
        fig.clear()


if __name__ == '__main__':
    unittest.main(exit=False)
