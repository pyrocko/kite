# KITE
## Introduction
Purpose of this framework is streamlining of InSAR processing routines for earthquake inversion through [pyrocko](http://www.pyrocko.org).

## Requirements and Installation

```sh
git clone https://gitext.gfz-potsdam.de/isken/kite.git
cd kite
sudo apt-get install python-pyside python-pyside.qtcore python-pyside.qtopengl
sudo pip install .
```

## Example
```python
from kite import Scene

scene = Scene.load('dataset.mat')  # load data from .mat file
scene.spool()  # start the GUI for data inspection and Quadtree parametrisation

# Inspection of covariance matrices for the quadtree
import matplotlib.pyplot as plt
cov_fast = scene.quadtree.covariance.matrix_fast
cov = scene.quadtree.covariance.matrix
plt.imshow(cov)
plt.show()
```
