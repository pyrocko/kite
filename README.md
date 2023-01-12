# Kite

[![Docs](https://img.shields.io/badge/kite-Documentation-blue.svg)](https://pyrocko.org/kite/docs/current/)
[![PyPI](https://img.shields.io/pypi/v/kite)](https://pypi.org/project/kite)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kite)
[![CI](https://github.com/pyrocko/kite/actions/workflows/build-wheels.yaml/badge.svg)](https://github.com/pyrocko/kite/actions/workflows/build-wheels.yaml)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_Preparation of InSAR surface displacement maps for geophysical modelling_

## Installation

Install from pip:

```sh
pip install kite
```

With additional gdal dependency, used for GeoTIFF (GACOS and LiCSAR):

```sh
pip install kite[gdal]
```

## Introduction

This framework is streamlining InSAR displacement processing routines for earthquake inversion through [Pyrocko](https://www.pyrocko.org) and Grond.

Kite features simple and efficient handling of displacement data:

* Import InSAR displacement data from GAMMA, ISCE, GMTSAR, ROI_PAC, SARScape, [COMET LiCSAR](https://comet.nerc.ac.uk/COMET-LiCS-portal/), [SNAP](https://step.esa.int/main/toolboxes/snap/) and Matlab
* Import of average timeseries from [STAMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/) and German [BodenBewegungsDienst](https://bodenbewegungsdienst.bgr.de)
* **Quadtree** calculation for data reduction
* **Covariance** estimation from data noise
* **APS removal** from [GACOS](http://ceg-research.ncl.ac.uk/v2/gacos/) atmoshperic models and empirical elevation correlation

## Citation

Recommended citation for Kite

> Isken, Marius; Sudhaus, Henriette; Heimann, Sebastian; Steinberg, Andreas; Daout, Simon; Vasyura-Bathke, Hannes (2017): Kite - Software for Rapid Earthquake Source Optimisation from InSAR Surface Displacement. V. 0.1. GFZ Data Services. <http://doi.org/10.5880/GFZ.2.1.2017.002>

[![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.1.2017.002-blue.svg)](http://doi.org/10.5880/GFZ.2.1.2017.002)

# Documentation

Find the documentation at <https://pyrocko.org/kite/docs/current/>.

## Short Example

```python
from kite import Scene

# Import Matlab container to kite
scene = Scene.load('SNAP_data/')
scene.spool()  # start the GUI for data inspection and Quadtree parametrisation
```

Visual parametrisation of the quadtree and spatial covariance for SLC and InSAR time-series.

![L'Aquila earthquake unwrapped displacement data](https://pyrocko.org/grond/docs/current/_images/example_spool-quadtree.png)
