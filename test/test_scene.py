#!/bin/python
import logging
import unittest
import numpy as num
from kite import TestScene, Scene

# from . import common

# common.setLogLevel('DEBUG')


def get_scene():
    sc = Scene()
    sc.frame.llLat = 52.395833
    sc.frame.llLon = 13.061389
    sc.frame.dE = .001
    sc.frame.dN = .001
    sc.frame.spacing = 'degree'
    sc.displacement = num.zeros((500, 500))

    return sc


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


def test_topo():
    logging.basicConfig(level=logging.DEBUG)
    sc = get_scene()
    return sc.get_elevation()


if __name__ == '__main__':
    test_topo()
