#!/bin/python
import unittest
import numpy as num
from common import Benchmark
from kite import Scene, SceneTest
import matplotlib.pyplot as plt
import os

benchmark = Benchmark()


class TestCovariance(unittest.TestCase):

    def setUp(self):
        file = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

        self.sc = Scene.import_file(file)
        self.sc.meta.scene_title = 'Matlab Input - Myanmar 2011-02-14'

        # self.sc = SceneTest.createGauss(ny=250)

    def __setUp(self):
        self.sc = SceneTest.createGauss()
        # self.sc._log.setLevel('CRITICAL')

    @unittest.skip('Skip!')
    def testCovariance(self):
        self.sc.quadtree.epsilon = .02
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance

        matrix_focal = cov.covariance_matrix
        matrix = cov.covariance_matrix_focal
        num.testing.assert_allclose(matrix, matrix_focal,
                                    rtol=1e-7, atol=2e3, verbose=True)

    @benchmark
    def testCovariancPool(self):
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance
        cov._calcDistanceMatrix(method='matrix', nthreads=4)

    @benchmark
    def testCovariancParallel(self):
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance
        cov._calcDistanceMatrix(method='matrix_c', nthreads=4)

    @benchmark
    def testCovariancSingle(self):
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance
        cov._calcDistanceMatrix(method='matrix_c', nthreads=1)

    @benchmark
    def testCovariancFocal(self):
        self.sc.quadtree.covariance.subsampling = 24
        cov = self.sc.quadtree.covariance
        cov._calcDistanceMatrix(method='focal')

    @unittest.skip('Skip!')
    def testCovarianceVisual(self):
        self.sc.quadtree.epsilon = .02
        cov = self.sc.quadtree.covariance
        cov.subsampling = 24
        # l = self.sc.quadtree.leafs[0]
        d = []
        # d.append(('Matrix - Pool', cov._calcDistanceMatrix(method='matrix')))
        d.append(('Matrix - C', cov._calcDistanceMatrix(method='matrix_c')))
        print d[-1][1]
        d.append(('Matrix - Focal', cov._calcDistanceMatrix(method='focal')))

        fig, _ = plt.subplots(1, len(d))
        for i, (title, mat) in enumerate(d):
            print '%s Max %f' % (title, num.nanmax(mat)), mat.shape
            fig.axes[i].imshow(mat)
            fig.axes[i].set_title(title)
        plt.show()


if __name__ == '__main__':
    unittest.main(exit=False)
    print benchmark
