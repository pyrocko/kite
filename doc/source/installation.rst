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
    
    git clone https://gitext.gfz-potsdam.de/isken/kite.git
    cd kite
    python setup.py --user --install-scripts=/usr/local/bin
