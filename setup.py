#!/bin/python
from setuptools import setup
from setuptools import find_packages

setup(
    name='kite',
    version='0.0.1',
    description='Handle SAR displacement data towards pyrocko',
    author='Marius P. Isken',
    author_email='misken@geophysik.uni-kiel.de',
    install_requires=['numpy>=1.9.0', 'pyrocko', 'scipy', 'pyyaml', 'progressbar'],
    packages=['kite', 'kite.spool'],
    package_dir={'kite': 'src'},
    data_files=[('kite/spool/ui/', ['src/spool/ui/spool.ui'])]
)
