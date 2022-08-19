from __future__ import absolute_import

import shutil
import tempfile
import unittest
from os import path as op

import numpy as num

from kite import Scene

from . import common

# format (dl_dir, load_file)
filenames = {
    "matlab": ("myanmar_alos_dsc_ionocorr.mat", None),
    "gmtsar": ("gmtsar/", "gmtsar/unwrap_ll.los_ll.grd"),
    "roi_pac": ("roi_pac/", "roi_pac/geo_20160113-20160206_atmo_2rlks_c10_cut.unw"),
    "gamma": ("gamma/", "gamma/asc"),
    "isce": ("isce/", "isce/filt_topophase.unw.geo"),
}

common.setLogLevel("DEBUG")


class SceneIOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="kite")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)


def _create_test_func(fmt, dl_path, filename):
    def f(self):
        common.get_test_data(dl_path)

        if filename is None:
            load_path = dl_path
        else:
            load_path = filename
        load_path = op.join(common.data_dir, load_path)

        fn_save = op.join(self.tmp_dir, "kite-%s" % fmt)

        sc1 = Scene.import_data(load_path)
        sc1.save(fn_save)

        sc2 = Scene.load(fn_save)

        num.testing.assert_equal(sc1.displacement, sc2.displacement)
        num.testing.assert_equal(sc1.phi, sc2.phi)
        num.testing.assert_equal(sc1.theta, sc2.theta)

    f.__name__ = "test_import_%s" % fmt

    return f


for fmt, fns in filenames.iteritems():
    setattr(SceneIOTest, "test_import_%s" % fmt, _create_test_func(fmt, *fns))
