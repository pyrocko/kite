# KITE
## Introduction
This framework is streamlining InSAR displacement processing routines for earthquake inversion through [pyrocko](http://www.pyrocko.org) and Grond.

Kite features simple and efficient handling of displacement data:

* Import InSAR displacement data from GAMMA, ISCE, GMTSAR and Matlab
* Efficient quadtree implementation
* Covariance estimation from noise
* Interactive GUI


## Installation and Requirement

### Requires libraries

* PySide with OpenGL support (Qt4)
* pyQtGraph
* NumPy
* SciPy
* pyyaml
* OpenMP
* pyrocko

Installation on Debian based distributions with `apt`

```sh
sudo apt-get install python-pyside python-pyside.qtcore python-pyside.qtopengl\
  python-yaml python-scipy python-numpy libomp-dev
```

### Native installation

```sh
git clone https://github.com/pyqtgraph/pyqtgraph.git
cd pyqtgraph
sudo python setup.py install
git clone https://github.com/pyrocko/kite.git
cd kite
sudo python setup.py install
```

### Installation through pip

```sh
sudo pip install git+https://github.com/pyqtgraph/pyqtgraph.git
sudo pip install git+https://github.com/pyrocko/kite.git
```

## Example
```python
from kite import Scene

# Import Matlab container to kite
scene = Scene.import_file('dataset.mat')
scene.spool()  # start the GUI for data inspection and Quadtree parametrisation

# Inspection of covariance parameters
scene.quadtree.covariance.plot()
```

## Documentation
Find the documentation at https://pyrocko.github.io/kite/.
