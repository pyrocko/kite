#!/bin/python
import unittest
import numpy as num
from kite.scene import Scene, SceneSynTest
from kite.quadtree import Quadtree


class TestGaussScene(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.sc = SceneSynTest.createGauss()

    def test_quadtree(self):
        qt = Quadtree(self.sc, .1)
        for e in num.linspace(0.118, .2, num=30):
            qt.epsilon = e

        for nan in num.linspace(0.1, 1., num=30):
            qt.nan_allowed = nan

        for s in num.linspace(100, 4000, num=30):
            qt.tile_size_lim = (s, 5000)

        for s in num.linspace(200, 4000, num=30):
            qt.tile_size_lim = (0, 5000)


class TestMatScene(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        _file = 'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat'

        self.sc = Scene.load(_file)
        self.sc.utm_x *= 1000
        self.sc.utm_y *= 1000
        self.sc.meta.title = 'Matlab Input - Myanmar 2011-02-14'

    def test_quadtree(self):
        qt = Quadtree(self.sc, .1)
        for e in num.linspace(0.118, .3, num=30):
            qt.epsilon = e

        for nan in num.linspace(0.1, 1., num=30):
            qt.nan_allowed = nan

        for s in num.linspace(100, 4000, num=30):
            qt.tile_size_lim = (s, 5000)

        for s in num.linspace(200, 4000, num=30):
            qt.tile_size_lim = (0, 5000)


if __name__ == '__main__':
    unittest.main()
