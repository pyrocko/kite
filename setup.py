#!/usr/bin/env python3
import os
import platform
import tempfile

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

version = '1.3.0'


def _have_openmp():
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
        print('Continuing your build using OpenMP...\n')
        return True

    import multiprocessing
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
    print('Continuing your build without OpenMP...\n')
    return False


if _have_openmp():

    omp_arg = ['-fopenmp']
    omp_lib = ['-lgomp']

    if platform.uname()[0] == 'Darwin':
        gomp_lib = os.environ.get('GOMPLIB', None)
        if gomp_lib:
            omp_lib.insert(0, '-L{}'.format(gomp_lib))
            omp_lib.append('-Wl,-rpath,{}'.format(gomp_lib))

        else:
            print('''Found the gcc compiler on MacOS but cannot find the
OpenMP libraries path at environment variable GOMPLIB.
''')
            print('Continuing your build without OpenMP...\n')
            omp_arg = []
            omp_lib = []

else:
    omp_arg = []
    omp_lib = []


setup(
    name='kite',
    version=version,
    description='Handle SAR displacement data towards pyrocko',
    author='Marius P. Isken, Henriette Sudhaus;'
           'BriDGes Emmily Noether-Programm (DFG)',
    author_email='misken@geophysik.uni-kiel.de',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    license='GPLv3',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: C',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        ],
    keywords=[
        'InSAR satellite radar earthquake optimization'],
    python_requires='>=3.5',
    packages=[
        'kite',
        'kite.util',
        'kite.sources',
        'kite.spool',
        'kite.talpa',
        'kite.talpa.sources'],
    package_dir={
        'kite': 'src'},
    package_data={
        'kite': ['spool/res/*',
                 'talpa/res/*']},
    entry_points={
        'console_scripts':
            ['spool = kite.spool.__main__:main',
             'talpa = kite.talpa.__main__:main',
             'stamps2kite = kite.util.stamps2kite:main',
             'bbd2kite = kite.util.bbd2kite:main']},
    ext_package='kite',
    ext_modules=[
        Extension(
            'covariance_ext',
            sources=[pjoin('src/ext', 'covariance.c')],
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
        Extension(
            'sources.disloc_ext',
            sources=[pjoin('src/sources/ext', 'disloc.c')],
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
