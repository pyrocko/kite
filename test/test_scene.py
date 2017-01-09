#!/bin/python
import unittest
import os.path
import numpy as num
from kite import Scene, SceneTest


class TestSyntheticsGenerators(unittest.TestCase):
    def testGauss(self):
        SceneTest.createGauss()

    def testFractal(self):
        SceneTest.createFractal()

    def testSine(self):
        SceneTest.createSine()


class TestSyntheticScene(unittest.TestCase):
    def setUp(self):
        self.sc = SceneTest.createGauss()
        self.sc.setLogLevel('ERROR')
        self.sc.quadtree.epsilon = .02
        self.sc.covariance.subsampling = 24

    def testQuadtree(self):
        qt = self.sc.quadtree
        for e in num.linspace(0.118, .2, num=30):
            qt.epsilon = e

        for nan in num.linspace(0.1, 1., num=30):
            qt.nan_allowed = nan

        for s in num.linspace(100, 4000, num=30):
            qt.tile_size_min = s
            qt.tile_size_max = 5000

        for s in num.linspace(200, 4000, num=30):
            qt.tile_size_min = 20
            qt.tile_size_max = s

    def testIO(self):
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp(prefix='kite')
        file = os.path.join(tmp_dir, self.__class__.__name__)
        sc1 = self.sc

        sc1.quadtree.epsilon = .120
        sc1.quadtree.tile_size_min = 50
        sc1.quadtree.tile_size_max = 23000
        sc1.quadtree.nan_allowed = .9
        try:
            sc1.save(file)
            sc2 = Scene()
            sc2.setLogLevel('ERROR')
            sc2.load(file)

            self.assertEqual(sc1.quadtree.epsilon,
                             sc2.quadtree.epsilon)
            self.assertEqual(sc1.quadtree.nan_allowed,
                             sc2.quadtree.nan_allowed)
            self.assertEqual(sc1.quadtree.tile_size_min,
                             sc2.quadtree.tile_size_min)
            self.assertEqual(sc1.quadtree.tile_size_max,
                             sc2.quadtree.tile_size_max)
            self.assertEqual(sc1.quadtree.nleafs,
                             sc2.quadtree.nleafs)
            self.assertEqual([l.id for l in sc1.quadtree.leafs],
                             [l.id for l in sc2.quadtree.leafs])

        finally:
            shutil.rmtree(tmp_dir)


class TestMatlabScene(unittest.TestCase):
    def setUp(self):
        file = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

        self.sc = Scene()
        self.sc.setLogLevel('ERROR')
        self.sc.import_data(file)
        self.sc.meta.scene_title = 'Matlab Input - Myanmar 2011-02-14'

    def testQuadtree(self):
        qt = self.sc.quadtree
        for e in num.linspace(0.118, .3, num=30):
            qt.epsilon = e

        for nan in num.linspace(0.1, 1., num=30):
            qt.nan_allowed = nan

        for s in num.linspace(100, 4000, num=30):
            qt.tile_size_min = s
            qt.tile_size_max = 5000

        for s in num.linspace(200, 4000, num=30):
            qt.tile_size_min = 0
            qt.tile_size_max = 5000

    def testIO(self):
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp(prefix='kite')
        # print(tmp_dir)
        file = os.path.join(tmp_dir, self.__class__.__name__)
        sc1 = self.sc

        sc1.quadtree.epsilon = .076
        sc1.quadtree.tile_size_min = 50
        sc1.quadtree.tile_size_max = 12773
        sc1.quadtree.nan_allowed = .8

        sc1.covariance.config.a = 0.008
        sc1.covariance.config.b = 300.2
        sc1.covariance.config.variance = .2
        sc1.covariance.covariance_matrix

        try:
            sc1.save(file)
            sc2 = Scene()
            sc2.setLogLevel('ERROR')
            sc2.load(file)

            self.assertEqual(sc1.quadtree.epsilon,
                             sc2.quadtree.epsilon)
            self.assertEqual(sc1.quadtree.nan_allowed,
                             sc2.quadtree.nan_allowed)
            self.assertEqual(sc1.quadtree.tile_size_min,
                             sc2.quadtree.tile_size_min)
            self.assertEqual(sc1.quadtree.tile_size_max,
                             sc2.quadtree.tile_size_max)
            self.assertEqual(sc1.quadtree.nleafs,
                             sc2.quadtree.nleafs)
            self.assertEqual([l.id for l in sc1.quadtree.leafs],
                             [l.id for l in sc2.quadtree.leafs])

            self.assertEqual(sc1.covariance.variance,
                             sc2.covariance.variance)
            self.assertEqual(sc1.covariance.covariance_model,
                             sc2.covariance.covariance_model)
            num.testing.assert_equal(sc1.covariance.weight_matrix_focal,
                                     sc2.covariance.weight_matrix_focal)
            num.testing.assert_equal(sc1.covariance.covariance_matrix_focal,
                                     sc2.covariance.covariance_matrix_focal)
            num.testing.assert_equal(sc1.covariance.covariance_matrix,
                                     sc2.covariance.covariance_matrix)

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()
