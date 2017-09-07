Installation
============
Kite has been tested on *Debian based distributions* (eg. Mint and Ubuntu).

Debian / Ubuntu
---------------
.. code-block :: sh

    sudo apt-get install python-pyside python-pyside.qtcore python-pyside.qtopengl python-yaml python-scipy python-numpy
    git clone https://gitext.gfz-potsdam.de/isken/kite.git
    cd kite
    sudo pip install .

MacOS (Sierra, MacPorts)
------------------------

Install without root, installing `pyrocko` is a prerequisite. Please visit http://pyrocko.org/v0.3/install_mac.html

.. code-block :: sh

	sudo port install git
	sudo port install python27
	sudo port select python python27
	sudo port install py27-numpy
	sudo port install py27-scipy
	sudo port install py27-matplotlib
	sudo port install py27-yaml
	sudo port install py27-pyside
	sudo port install py27-setuptools
	/opt/local/bin/easy_install-2.7 --user pyavl	
	 
    git clone https://gitext.gfz-potsdam.de/isken/kite.git
    cd kite
    python setup.py install --user --install-scripts=/usr/local/bin
