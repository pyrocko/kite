#!/usr/bin/env python3
import unittest

import matplotlib.pyplot as plt
import numpy as num

from kite import Scene

from . import common

benchmark = common.Benchmark()
common.setLogLevel("DEBUG")


class TestCovariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = common.get_test_data("myanmar_alos_dsc_ionocorr.mat")
        cls.sc = Scene.import_data(file)

    def test_covariance(self):
        cov = self.sc.covariance
        cov.epsilon = 0.02
        cov.subsampling = 24

        d = []
        d.append(("Full", cov._calcCovarianceMatrix(method="full", nthreads=0)))
        d.append(("Focal", cov._calcCovarianceMatrix(method="focal")))

        for _, c1 in d:
            for _, c2 in d:
                num.testing.assert_allclose(c1, c2, rtol=200, atol=2e3, verbose=True)

    def test_synthetic_noise(self):
        self.sc.covariance.syntheticNoise()
        self.sc.covariance.variance

    def test_quadtree_noise(self):
        rstate = num.random.RandomState()
        self.sc.covariance.getQuadtreeNoise(rstate=rstate)

    def test_covariance_parallel(self):
        self.sc.quadtree.epsilon = 0.07
        self.sc.quadtree.tile_size_max = 11000

        cov = self.sc.covariance
        cov.config.adaptive_subsampling = True

        @benchmark
        def calc_exp():
            return cov._calcCovarianceMatrix(method="full", nthreads=0)

        @benchmark
        def calc_exp_cos():
            cov.setModelFunction("exponential_cosine")
            return cov._calcCovarianceMatrix(method="full", nthreads=0)

        res = calc_exp()
        ref = num.load("test/covariance_ref.npy")
        # calc_exp_cos()
        num.testing.assert_array_equal(ref, res)
        print(benchmark)

    @benchmark
    def _test_covariance_single_thread(self):
        cov = self.sc.covariance
        cov._calcCovarianceMatrix(method="full", nthreads=1)

    @benchmark
    def test_covariance_focal(self):
        cov = self.sc.covariance
        cov._calcCovarianceMatrix(method="focal")

    @unittest.skip("Skip!")
    def _test_covariance_visual(self):
        cov = self.sc.covariance
        cov.epsilon = 0.02
        cov.subsampling = 10
        # l = self.sc.quadtree.leaves[0]
        d = []
        d.append(("Full", cov._calcCovarianceMatrix(method="full", nthreads=0)))
        d.append(("Focal", cov._calcCovarianceMatrix(method="focal")))

        fig, _ = plt.subplots(1, len(d))
        for i, (title, mat) in enumerate(d):
            print("%s Max %f" % ((title, num.nanmax(mat)), mat.shape))
            fig.axes[i].imshow(mat)
            fig.axes[i].set_title(title)
        plt.show()

    def test_covariance_spatial(self):
        cov = self.sc.covariance
        quad = self.sc.quadtree  # noqa

        @benchmark
        def calc(c):
            cov, dist = c.covariance_spatial
            # assert num.all(num.isfinite(cov))

        for i in range(10):
            calc(cov)

        print(benchmark)


if __name__ == "__main__":
    unittest.main(exit=False)
    print(benchmark)
