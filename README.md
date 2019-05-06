# KITE

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)

## Introduction
This framework is streamlining InSAR displacement processing routines for earthquake inversion through [pyrocko](https://www.pyrocko.org) and Grond.

Kite features simple and efficient handling of displacement data:

* Import InSAR displacement data from GAMMA, ISCE, GMTSAR, ROI_PAC, SARScape, [COMET LiCSAR](https://comet.nerc.ac.uk/COMET-LiCS-portal/) and Matlab
* Efficient **quadtree** implementation
* **Covariance** estimation from noise
* **Interactive GUI**

# Documentation
Find the documentation at https://pyrocko.org/kite/docs/.

## Short Example
```python
from kite import Scene

# Import Matlab container to kite
scene = Scene.import_file('dataset.mat')
scene.spool()  # start the GUI for data inspection and Quadtree parametrisation

# Inspection of covariance parameters
scene.quadtree.covariance.plot()
```

## Installation and Requirement

### Requires libraries

* PyQt5 with OpenGL support
* PyQtGraph
* NumPy
* SciPy

Installation on Debian based distributions through `apt`

```sh
sudo apt-get install python-pyside python-pyside.qtcore python-pyside.qtopengl\
  python-yaml python-scipy python-numpy
```

### Installation through pip

```sh
sudo pip install git+https://github.com/pyqtgraph/pyqtgraph.git
sudo pip install git+https://github.com/pyrocko/kite.git
```

### Native installation

```sh
git clone https://github.com/pyqtgraph/pyqtgraph.git
cd pyqtgraph; sudo python setup.py install
git clone https://github.com/pyrocko/kite.git
cd kite; sudo python setup.py install
```

## Citation
Recommended citation for Kite

> Isken, Marius; Sudhaus, Henriette; Heimann, Sebastian; Steinberg, Andreas; Daout, Simon; Vasyura-Bathke, Hannes (2017): Kite - Software for Rapid Earthquake Source Optimisation from InSAR Surface Displacement. V. 0.1. GFZ Data Services. http://doi.org/10.5880/GFZ.2.1.2017.002

[![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.1.2017.002-blue.svg)](http://doi.org/10.5880/GFZ.2.1.2017.002)
