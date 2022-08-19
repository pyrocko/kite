Installation
============

Kite is written for `Python 3 <https://python.org>`_, the installation has been tested on Debian based distributions (e.g. Ubuntu and Mint), MacOSX, and `Anaconda <https://anaconda.org/pyrocko/kite>`_.

Debian / Ubuntu
---------------

As a mandatory prerequisite you have to install Pyrocko, visit `Pyrocko installation instructions <https://pyrocko.org/docs/current/install/index.html>`_ for details.

.. code-block :: sh
    :caption: Installation from source and ``apt-get``

    sudo apt-get install python3-dev python3-pyqt5 python3-pyqt5 python3-pyqt5.qtopengl python3-scipy python3-numpy python3-pyqtgraph

    git clone https://github.com/Turbo87/utm
    cd utm
    sudo python3 setup.py install

    git clone https://github.com/pyrocko/kite
    cd kite
    sudo python3 setup.py install


PIP
---

An installation through ``pip`` requires the same prerequisites as above:

.. code-block :: sh
    :caption: Installation through pip

    sudo pip3 install utm
    sudo pip3 install git+https://github.com/pyrocko/kite


MacOS (Sierra, MacPorts)
------------------------

Installing Pyrocko is a prerequisite, visit `Pyrocko Mac installation instructions <http://pyrocko.org/docs/current/install_mac.html>`_ for details.

Kite installation instructions with `MacPorts <https://www.macports.org/>`_, alternatively check out the Anaconda3 instructions:

.. code-block :: sh
    :caption: Installation from source through MacPorts

    sudo port install git
    sudo port install python35
    sudo port select python python35
    sudo port install py35-numpy
    sudo port install py35-scipy
    sudo port install py35-matplotlib
    sudo port install py35-yaml
    sudo port install py35-pyqt5
    sudo port install py35-setuptools
    sudo pip3 install git+https://github.com/Turbo87/utm
    sudo pip3 install git+https://github.com/pyqtgraph/pyqtgraph

    git clone https://github.com/pyrocko/kite
    cd kite
    python3 setup.py install --user --install-scripts=/usr/local/bin


Anaconda3 using ``pip``
--------------------------

Use ``pip`` on Anaconda to install the software:

.. code-block:: sh
    :caption: Install the prerequisites through ``conda``

    pip install git+https://github.com/Turbo87/utm
    pip install git+https://github.com/pyrocko/kite
