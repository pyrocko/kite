from __future__ import absolute_import

import unittest
import tempfile
import shutil
import os
import numpy as num

from . import common
from kite import Scene

filenames = {
    'matlab': 'myanmar_alos_dsc_ionocorr.mat',
    'gmtsar': 'gmtsar/',
}


class SceneIOTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix='kite')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)


def _make_function(fmt, filename):
    def f(self):
        fn = common.get_test_data(filename)
        if isinstance(fn, list):
            fn = fn[-1]

        fn_save = os.path.join(self.tmp_dir, 'kite-%s' % fmt)

        sc1 = Scene.import_data(fn)
        sc1.save(fn_save)

        sc2 = Scene.load(fn_save)

        num.testing.assert_equal(sc1.displacement, sc2.displacement)
        num.testing.assert_equal(sc1.phi, sc2.phi)
        num.testing.assert_equal(sc1.theta, sc2.theta)

    f.__name__ = 'test_import_%s' % fmt

    return f


for fmt, filename in filenames.iteritems():
    setattr(
        SceneIOTest,
        'test_import_%s' % fmt,
        _make_function(fmt, filename))
