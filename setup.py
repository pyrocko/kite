#!/bin/python
from setuptools import setup, Extension
from os.path import join as pjoin
import time
try:
    import numpy
except ImportError:
    class numpy():
        def __init__(self):
            pass

        @classmethod
        def get_include(self):
            return ''


def xcode_version_str():
    from subprocess import Popen, PIPE
    try:
        version = Popen(['xcodebuild', '-version'], stdout=PIPE, shell=False)\
            .communicate()[0].split()[1]
    except IndexError:
        version = None
    return version


def support_omp():
    import platform
    from distutils.version import StrictVersion
    if platform.mac_ver() == ('', ('', '', ''), ''):
        return True
    else:
        v_string = xcode_version_str()
        if v_string is None:
            return False
        else:
            v = StrictVersion(v_string)
            return v < StrictVersion('4.2.0')


if support_omp():
    omp_arg = ['-fopenmp']
    omp_lib = ['-lgomp']
    print('OpenMP found')
else:
    omp_arg = []
    omp_lib = []
    print('OpenMP not found')

setup(
    name='kite',
    version='0.0.2.post%s' % time.strftime('%Y%m%d'),
    description='Handle SAR displacement data towards pyrocko',
    author='Marius P. Isken, Henriette Sudhaus;'
           'BRiDGES Emmily Noether-Programm (DFG)',
    author_email='misken@geophysik.uni-kiel.de',

    license='GPL',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: C',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        ],
    keywords=['insar satellite radar earthquake optimization'],

    install_requires=['numpy>=1.9.0', 'pyrocko', 'scipy', 'pyyaml',
                      'progressbar', 'utm', 'pyqtgraph>=0.10.0'],
    packages=['kite', 'kite.spool'],
    package_dir={'kite': 'src'},
    data_files=[('kite/spool/ui/', ['src/spool/ui/spool.ui',
                                    'src/spool/ui/about.ui',
                                    'src/spool/ui/logging.ui',
                                    'src/spool/ui/transect.ui',
                                    'src/spool/ui/noise_dialog.ui',
                                    'src/spool/ui/covariance_matrix.ui',
                                    'src/spool/ui/boxkite-sketch.jpg',
                                    'src/spool/ui/radar_splash.png'])],
    entry_points={
        'console_scripts': ['spool = kite.spool.__main__:main'],
    },

    ext_modules=[
        Extension('covariance_ext',
                  sources=[pjoin('src/ext', 'covariance_ext.c')],
                  include_dirs=[numpy.get_include()],
                  define_macros=None,
                  undef_macros=None,
                  library_dirs=None,
                  libraries=None,
                  runtime_library_dirs=None,
                  extra_objects=None,
                  extra_compile_args=[] + omp_arg,
                  extra_link_args=[] + omp_lib,
                  export_symbols=None,
                  swig_opts=None,
                  depends=None,
                  language=None)
    ]
)
