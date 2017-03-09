#!/bin/python
import os
import tempfile
import time

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


def _check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from pynbody // yt.
    Thanks to Nathan Goldbaum and Andrew Pontzen.
    """
    import distutils.sysconfig
    import subprocess
    import shutil

    tmpdir = tempfile.mkdtemp(prefix='kite')
    compiler = os.environ.get(
      'CC', distutils.sysconfig.get_config_var('CC')).split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    tmpfile = pjoin(tmpdir, 'check_openmp.c')
    with open(tmpfile, 'w') as f:
        f.write('''
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d", omp_get_thread_num());
}
''')

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', '-o%s'
                                         % pjoin(tmpdir, 'check_openmp'),
                                         tmpfile],
                                        stdout=fnull, stderr=fnull)
    except OSError:
        exit_code = 1
    finally:
        shutil.rmtree(tmpdir)

    if exit_code == 0:
        print ('Continuing your build using OpenMP...\n')
        return True

    import multiprocessing
    import platform
    if multiprocessing.cpu_count() > 1:
        print('''WARNING
OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.
Some routines in kite are parallelized using OpenMP and these will
only run on one core with your current configuration.
''')
        if platform.uname()[0] == 'Darwin':
            print('''Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -
sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build
''')
    print ('Continuing your build without OpenMP...\n')
    return False


if _check_for_openmp():
    omp_arg = ['-fopenmp']
    omp_lib = ['-lgomp']
else:
    omp_arg = []
    omp_lib = []

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
    package_data={'kite': ['spool/ui/*']},
    entry_points={
        'console_scripts': ['spool = kite.spool.__main__:main'],
    },

    ext_package='kite',
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
                  language=None),
    ]
)
