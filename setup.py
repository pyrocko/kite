#!/bin/python
from setuptools import setup, Extension
from os.path import join as pjoin
try:
    import numpy
except ImportError:
    class numpy():
        def __init__(self):
            pass

        @classmethod
        def get_include(self):
            return ''
import time

setup(
    name='kite',
    version='0.0.2-%s' % time.strftime('%Y%m%d'),
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
                                    'src/spool/ui/boxkite-sketch.jpg'])],
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
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-lgomp'],
                  export_symbols=None,
                  swig_opts=None,
                  depends=None,
                  language=None)
    ]
)
