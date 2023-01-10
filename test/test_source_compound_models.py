import unittest

import numpy as np

from kite import SandboxScene
from kite.sources import EllipsoidSource, PointCompoundSource
from kite.sources import compound_engine as cm  # noqa

from . import common

km = 1e3
plot = False

benchmark = common.Benchmark()
common.setLogLevel("DEBUG")


class CompoundModelsTest(unittest.TestCase):
    def test_ECM(self):
        nrows = 500
        ncols = 500

        x0 = 250.0
        y0 = 250.0
        depth = 30.0

        rotx = 0.0
        roty = 0.0
        rotz = 0.0

        ax = 1.0
        ay = 1.0
        az = 0.25

        # ax = 1.
        # ay = 1.
        # az = 1.

        P = 1e6
        mu = 0.33e11
        lamda = 0.33e11

        X, Y = np.meshgrid(np.arange(nrows), np.arange(ncols))

        coords = np.empty((nrows * ncols, 2))
        coords[:, 0] = X.ravel()
        coords[:, 1] = Y.ravel()

        # Testing shapeTensor sub-functions
        nu = lamda / (lamda + mu) / 2
        cm.shapeTensor(ax, ay, az, nu)

        @benchmark
        def runECM():
            return cm.ECM(
                coords, x0, y0, depth, rotx, roty, rotz, ax, ay, az, P, mu, lamda
            )

        ue, un, uv, _, _ = runECM()

        self._plot_displacement(un.reshape(nrows, ncols))

    def _test_ECM_against_Octave(self):
        # Lost reference file
        from os import path as p

        from scipy import io

        X, Y = np.meshgrid(np.linspace(-7.0, 7.0, 701), np.linspace(-5.0, 5.0, 501))
        x0 = 0.5
        y0 = -0.25
        depth = 2.75

        rotx = 5.0
        roty = -8.0
        rotz = 30

        ax = 1.0
        ay = 0.75
        az = 0.25

        P = 1e6
        mu = 0.33e11
        lamda = 0.33e11

        coords = np.empty((X.size, 2))
        coords[:, 0] = X.ravel()
        coords[:, 1] = Y.ravel()

        @benchmark
        def runECM():
            return cm.ECM(
                coords, x0, y0, depth, rotx, roty, rotz, ax, ay, az, P, mu, lamda
            )

        ue, un, uv, _, _ = runECM()

        ue = ue.reshape(*X.shape)
        un = un.reshape(*X.shape)
        uv = uv.reshape(*X.shape)

        mat = io.loadmat(
            p.join(p.dirname(__file__), "data", "displacement_ellipsoid_octave.mat")
        )

        np.testing.assert_equal(X, mat["X"])
        np.testing.assert_equal(Y, mat["Y"])

        for pym, comp in zip([ue, un, uv], ["ue", "un", "uv"]):
            m = mat[comp]
            # print([pym.min(), pym.max()], [m.min(), m.max()])
            np.testing.assert_allclose(pym, m, rtol=1e-11)

        self._plot_displacement(uv)
        self._plot_displacement(mat["uv"])

    def testEllipsoidSource(self):
        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        ms = SandboxScene()
        src = EllipsoidSource(
            easting=r(0.0, ms.frame.E.max()),
            northing=r(0.0, ms.frame.N.max()),
            depth=1e3,
        )
        src.regularize()

        ms.addSource(src)

        self._plot_modelScene(ms)

    def _test_pointCDM_against_Octave(self):
        from os import path as p

        from scipy import io

        X, Y = np.meshgrid(np.linspace(-7.0, 7.0, 701), np.linspace(-5.0, 5.0, 501))
        x0 = 0.5
        y0 = -0.25
        depth = 2.75

        rotx = 5.0
        roty = -8.0
        rotz = 30.0

        dVx = 0.00144
        dVy = 0.00128
        dVz = 0.00072

        nu = 0.25

        coords = np.empty((X.size, 2))
        coords[:, 0] = X.ravel()
        coords[:, 1] = Y.ravel()

        @benchmark
        def run_pointCDM():
            return cm.pointCDM(
                coords, x0, y0, depth, rotx, roty, rotz, dVx, dVy, dVz, nu
            )

        ue, un, uv = run_pointCDM()

        ue = ue.reshape(*X.shape)
        un = un.reshape(*X.shape)
        uv = uv.reshape(*X.shape)

        mat = io.loadmat(
            p.join(p.dirname(__file__), "data", "displacement_pcdm_octave.mat")
        )

        np.testing.assert_equal(X, mat["X"])
        np.testing.assert_equal(Y, mat["Y"])

        for pym, comp in zip([ue, un, uv], ["ue", "un", "uv"]):
            m = mat[comp]
            # print([pym.min(), pym.max()], [m.min(), m.max()])
            np.testing.assert_allclose(pym, m, rtol=1e-9)

        self._plot_displacement(mat["uv"])
        self._plot_displacement(uv)

    def testPointCompoundSourceSource(self):
        def r(lo, hi):
            return np.random.randint(lo, high=hi, size=1).astype(float)

        ms = SandboxScene()
        src = PointCompoundSource(
            easting=r(0.0, ms.frame.E.max()),
            northing=r(0.0, ms.frame.N.max()),
            depth=1e3,
        )
        src.regularize()
        ms.addSource(src)

        self._plot_modelScene(ms)

    @staticmethod
    def _plot_modelScene(ms):
        if not plot:
            ms.down
            return

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ms.processSources()

        ax.imshow(
            np.flipud(ms.down),
            aspect="equal",
            extent=[0, ms.frame.E.max(), 0, ms.frame.N.max()],
        )
        plt.show()

    @staticmethod
    def _plot_displacement(u):
        if not plot:
            u
            return

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()

        ax.imshow(u)
        plt.show()


if __name__ == "__main__":
    plot = True
    unittest.main(exit=False)
    print(benchmark)
