import unittest
import numpy as num

from kite import ModelScene
from kite.mod_disloc import OkadaSource, OkadaPlane


class testOkada(unittest.TestCase):
    def setUp(self):
        self.ms = ModelScene()

    def testOkadaSource(self):
        nsources = 1

        def r(lo, hi):
            return num.random.randint(lo, high=hi, size=1).astype(num.float)

        for s in xrange(nsources):
            self.ms.addSource(
                OkadaSource(
                    easting=r(0., self.ms.frame.E.max()),  # ok
                    northing=r(0., self.ms.frame.N.max()),  # ok
                    depth=r(0, 8000),  # ok
                    strike=r(0, 360),  # ok
                    dip=r(0, 170),
                    slip=r(1, 5),  # ok
                    rake=r(0, 180),
                    width=r(3000, 5000),
                    length=r(5000, 15000)))

        # self.plotDisplacement(self.ms)

    @staticmethod
    def plotDisplacement(ms):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        fig = plt.figure()
        ax = fig.gca()

        ax.imshow(num.flipud(ms.displacement), aspect='equal',
                  extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()])
        for src in ms.sources:
            p = Polygon(src.outline(), alpha=.8, fill=False)
            ax.add_artist(p)

        plt.show()


if __name__ == '__main__':
    unittest.main(exit=False)
