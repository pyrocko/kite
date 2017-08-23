#!/bin/python
import unittest
import os.path
import numpy as num
from kite import Scene, TestScene


class TestSyntheticsGenerators(unittest.TestCase):
    def testGauss(self):
        TestScene.createGauss()

    def testFractal(self):
        TestScene.createFractal()

    def testSine(self):
        TestScene.createSine()


class TestSyntheticScene(unittest.TestCase):
    def setUp(self):
        self.sc = TestScene.createGauss()
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


if __name__ == '__main__':
    unittest.main()
