import unittest
import time
import numpy as num

from kite import ModelScene
from kite.mod_disloc import OkadaSource, OkadaPath


class testOkada(unittest.TestCase):
    def setUp(self):
        self.ms = ModelScene()

    def testOkadaSource(self):
        nsources = 2

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            length = r(5000, 15000)
            self.ms.addSource(
                OkadaSource(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    slip=r(1, 5),  # ok
                    rake=r(0, 180),
                    length=length,
                    width=15. * length**.66,))

        # self.plotDisplacement(self.ms)

    def testOkadaPath(self):
        path = OkadaPath(
            origin_easting=10000,
            origin_northing=24000,)
        path.addNode(15000, 28000)
        path.addNode(18000, 32000)
        path.addNode(22000, 34000)
        path.insertNode(1, 22000, 34000)
        self.ms.addSource(path)

        self.plotDisplacement(self.ms)

    @staticmethod
    def plotDisplacement(ms):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        fig = plt.figure()
        ax = fig.gca()

        ax.imshow(num.flipud(ms.displacement), aspect='equal',
                  extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()])
        for src in ms.sources:
            for seg in src.segments:
                p = Polygon(seg.outline(), alpha=.8, fill=False)
                ax.add_artist(p)
            if isinstance(src, OkadaPath):
                nodes = num.array(src.nodes)
                ax.scatter(nodes[:, 0], nodes[:, 1], color='r')
        plt.show()
        fig.clear()


if __name__ == '__main__':
    unittest.main(exit=False)
