#!/bin/python
import unittest
import os
import numpy as num
from kite.scene import Scene, SceneSynTest
from kite.quadtree import Quadtree

_ifig = 0
_fig_dir = '/tmp/unittest_kite_figures/'


def _save_fig(fig):
    global _ifig
    try:
        os.mkdir(_fig_dir)
    except OSError:
        pass

    fig.savefig('%s/test_figure%02d.png' % (_fig_dir, _ifig))
    fig.clear()
    _ifig += 1


class TestGaussScene(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGaussScene, self).__init__(*args, **kwargs)
        self.sar = SceneSynTest.createGauss()

    def test_cartesian_conversion(self):
        r = num.sqrt(self.sar.cartesian.dx**2 +
                     self.sar.cartesian.dy**2 + self.sar.cartesian.dz**2)

        # theta = num.arccos(self.sar.cartesian.dz / r)
        # phi = num.arctan(self.sar.cartesian.dy / self.sar.cartesian.dx)

        num.testing.assert_almost_equal(self.sar.los.displacement, r)
        # num.testing.assert_almost_equal(self.sar.los.theta, theta)
        # num.testing.assert_almost_equal(self.sar.los.phi, phi)
        num.testing.assert_almost_equal(self.sar.los.displacement,
                                        self.sar.cartesian.dr)

    def test_plotting(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for comp in ['displacement']:
            self.sar.los.plot(comp, figure=fig)
            _save_fig(fig)

        for comp in ['theta', 'phi']:
            im = self.sar.los.plot(comp, figure=fig, cmap='spectral')
            im.set_clim(im.get_array().min(), im.get_array().max())
            _save_fig(fig)

        for comp in ['dr', 'dx', 'dy', 'dz']:
            self.sar.cartesian.plot(comp, figure=fig)
            _save_fig(fig)

    def test_quadtree(self):
        qt = Quadtree(self.sar, .1)
        for e in num.linspace(0.01, .1, num=30):
            qt.epsilon = e

        for e in num.linspace(0.1, .01, num=30):
            qt.epsilon = e


class TestMatScene(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMatScene, self).__init__(*args, **kwargs)

        import scipy.io as io
        import os

        _file = 'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat'
        mat = io.loadmat(os.path.join(os.path.dirname(__file__), _file))

        self.sar = Scene()
        self.sar.meta.title = 'Matlab Input - Myanmar 2011-02-14'

        self.sar.los.phi = mat['phi_dsc_defo']
        self.sar.los.theta = mat['theta_dsc_defo']
        self.sar.los.displacement = mat['ig_dc']
        self.sar.x = mat['xx_ig']
        self.sar.y = mat['yy_ig']

    def test_cartesian_conversion(self):
        # theta = num.arccos(self.sar.cartesian.dz / r)
        # phi = num.arctan(self.sar.cartesian.dy / self.sar.cartesian.dx)

        # num.testing.assert_almost_equal(self.sar.los.theta, theta)
        # num.testing.assert_almost_equal(self.sar.los.phi, phi)
        num.testing.assert_almost_equal(self.sar.los.displacement,
                                        self.sar.cartesian.dr)

    def test_plotting(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for comp in ['displacement']:
            self.sar.los.plot(comp, figure=fig)
            _save_fig(fig)

        for comp in ['theta', 'phi']:
            im = self.sar.los.plot(comp, figure=fig, cmap='spectral')
            im.set_clim(im.get_array().min(), im.get_array().max())
            _save_fig(fig)

        for comp in ['dr', 'dx', 'dy', 'dz']:
            self.sar.cartesian.plot(comp, figure=fig)
            _save_fig(fig)

    def test_quadtree(self):
        qt = Quadtree(self.sar, .1)
        for e in num.linspace(0.01, .1, num=30):
            qt.epsilon = e

        for e in num.linspace(0.1, .01, num=30):
            qt.epsilon = e


if __name__ == '__main__':
    unittest.main()
