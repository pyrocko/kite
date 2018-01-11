Installation
============

Kite has been tested on Anaconda, *Debian based distributions* (eg. Ubuntu and Mint) and MacOSX.

Anaconda3 using ``conda``
--------------------------

Pre-built packages are available for Linux 64-Bit and MacOS. You can use can use the ``conda`` package manager to install Kite:

.. code-block:: sh
    :caption: Packages are available on Anaconda Cloud

    conda install -c pyrocko pyrocko-kite


Debian / Ubuntu
---------------

As a mandatory prerequisit you have to install ``pyrocko`` - `Installation Instructions <https://pyrocko.org/docs/current/install/index.html>`_

.. code-block :: sh
    :caption: Installation through from source and ``apt``

    sudo apt-get install python3-dev python3-pyqt5 python3-pyqt5 python3-pyqt5.qtopengl python3-scipy python3-numpy
    git clone https://github.com/pyrocko/kite
    cd kite
    sudo python3 setup.py install

MacOS (Sierra, MacPorts)
------------------------

Install without root, installing `pyrocko` is a prerequisite. Please visit http://pyrocko.org/v0.3/install_mac.html

.. code-block :: sh

	sudo port install git
	sudo port install python35
	sudo port select python python35
	sudo port install py35-numpy
	sudo port install py35-scipy
	sudo port install py35-matplotlib
	sudo port install py35-yaml
	sudo port install py35-pyqt5
	sudo port install py35-setuptools
	 
    git clone https://gitext.gfz-potsdam.de/isken/kite.git
    cd kite
    python setup.py install --user --install-scripts=/usr/local/bin
