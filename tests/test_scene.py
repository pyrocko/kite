#!/bin/python
import unittest
import os.path
import numpy as num
from kite import Scene, SceneTest


class TestGaussScene(unittest.TestCase):
    def setUp(self):
        self.sc = SceneTest.createGauss()

    def testQuadtree(self):
        qt = self.sc.quadtree
        for e in num.linspace(0.118, .2, num=30):
            qt.epsilon = e

        for nan in num.linspace(0.1, 1., num=30):
            qt.nan_allowed = nan

        for s in num.linspace(100, 4000, num=30):
            qt.tile_size_lim = (s, 5000)

        for s in num.linspace(200, 4000, num=30):
            qt.tile_size_lim = (0, 5000)

    def testCovariance(self):
        self.sc.quadtree.epsilon = .118
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance

        matrix_focal = cov.covariance_matrix
        matrix = cov.covariance_matrix_focal
        num.testing.assert_allclose(matrix, matrix_focal,
                                    rtol=1e-7, atol=2e3, verbose=True)

    def testIO(self):
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp()
        try:
            self.sc.save(os.path.join(tmp_dir,
                         self.__class__.__name__))
            load_scene = Scene.load(os.path.join(tmp_dir,
                                                 self.__class__.__name__))
            del load_scene
        finally:
            pass
            shutil.rmtree(tmp_dir)


class TestMatlabScene(unittest.TestCase):

    def setUp(self):
        file = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

        self.sc = Scene.import_file(file)
        self.sc.meta.scene_title = 'Matlab Input - Myanmar 2011-02-14'

    def testQuadtree(self):
        qt = self.sc.quadtree
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
