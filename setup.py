#!/bin/python
from setuptools import setup
from setuptools import find_packages

setup(
    name='kite',
    version='0.0.1',
    description='Handle SAR displacement data towards pyrocko',
    author='Marius P. Isken',
    author_email='misken@geophysik.uni-kiel.de',
    install_requires=['numpy', 'logging', 'pyrocko', 'scipy'],
    packages=['kite'],
    package_dir={'kite': 'src'},
)
