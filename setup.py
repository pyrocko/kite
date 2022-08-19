#!/usr/bin/env python3
import os
import platform
import tempfile
from distutils.sysconfig import get_python_inc
from os.path import join as pjoin

from setuptools import Extension, setup

try:
    import numpy
except ImportError:

    class numpy:
        def __init__(self):
            ...

        @classmethod
        def get_include(cls):
            return ""


def _have_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from pynbody // yt.
    Thanks to Nathan Goldbaum and Andrew Pontzen.
    """
    import distutils.sysconfig
    import shutil
    import subprocess

    tmpdir = tempfile.mkdtemp(prefix="kite")
    compiler = os.environ.get("CC", distutils.sysconfig.get_config_var("CC")).split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    tmpfile = pjoin(tmpdir, "check_openmp.c")
    with open(tmpfile, "w") as f:
        f.write(
            """
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d", omp_get_thread_num());
}
"""
        )

    try:
        with open(os.devnull, "w") as fnull:
            exit_code = subprocess.call(
                [compiler, "-fopenmp", "-o%s" % pjoin(tmpdir, "check_openmp"), tmpfile],
                stdout=fnull,
                stderr=fnull,
            )
    except OSError:
        exit_code = 1
    finally:
        shutil.rmtree(tmpdir)

    if exit_code == 0:
        print("Continuing your build using OpenMP...\n")
        return True

    import multiprocessing

    if multiprocessing.cpu_count() > 1:
        print(
            """WARNING
OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.
Some routines in kite are parallelized using OpenMP and these will
only run on one core with your current configuration.
"""
        )
        if platform.uname()[0] == "Darwin":
            print(
                """Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -
sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build
"""
            )
    print("Continuing your build without OpenMP...\n")
    return False


if _have_openmp():

    omp_arg = ["-fopenmp"]
    omp_lib = ["-lgomp"]

    if platform.uname()[0] == "Darwin":
        gomp_lib = os.environ.get("GOMPLIB", None)
        if gomp_lib:
            omp_lib.insert(0, "-L{}".format(gomp_lib))
            omp_lib.append("-Wl,-rpath,{}".format(gomp_lib))

        else:
            print(
                """Found the gcc compiler on MacOS but cannot find the
OpenMP libraries path at environment variable GOMPLIB.
"""
            )
            print("Continuing your build without OpenMP...\n")
            omp_arg = []
            omp_lib = []

else:
    omp_arg = []
    omp_lib = []


setup(
    ext_modules=[
        Extension(
            "kite.covariance_ext",
            sources=[pjoin("kite/ext", "covariance.c")],
            include_dirs=[numpy.get_include(), get_python_inc()],
            extra_compile_args=omp_arg,
            extra_link_args=omp_lib,
            language="c",
        ),
        Extension(
            "kite.sources.disloc_ext",
            sources=[pjoin("kite/sources/ext", "disloc.c")],
            include_dirs=[numpy.get_include(), get_python_inc()],
            extra_compile_args=omp_arg,
            extra_link_args=omp_lib,
            language="c",
        ),
    ]
)
