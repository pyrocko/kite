Quickstart
==========

Kite supports importing unwrapped displacement scenes from different InSAR processors:

* `GMT5SAR <https://gmt.soest.hawaii.edu/projects/gmt5sar>`_
* `GAMMA Software <http://www.gamma-rs.ch/no_cache/software.html>`_
* `ROI_PAC <http://www.geo.cornell.edu/eas/PeoplePlaces/Faculty/matt/roi_pac.html/>`_
* `ISCE <https://winsar.unavco.org/software/isce>`_
* `SARscape <http://www.sarmap.ch/page.php?page=sarscape>`_
* `LiCSAR <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_


Each processor delivers different file formats and metadata. In order to import the data into Kite, data has to be prepared. Details for each format are described in :mod:`kite.scene_io`.


Import InSAR displacement
-------------------------

We will start with importing a scene from GMT5SAR.

.. code-block :: python
    :caption: GMT5SAR is an open-source processor based on GMT. We will import a binary ``.grd`` file.

    from kite import Scene

    # We import a unwrapped interferrogram scene.
    # The format shall be detected automatically
    # in this case processed a GMTSAR5

    sc = Scene.import_data('unwrap_ll.grd')
    sc.spool()


Download and Load Data from COMET LiCSAR
----------------------------------------

A slim downloader for COMET LiCSAR products is included in `kite.clients`. The script will download the passed unwrapped LiCSAR data and necessary LOS geotiffs into the current directory.

This example will download data from the 2017 Iran–Iraq earthquake (M 7.3) from the `COMET LiCSAR Portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_:


.. code-block :: sh

    python3 -m kite.clients http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/6/006D_05509_131313/products/20171107_20171201/20171107_20171201.geo.unw.tif .


To open the scene in spool, run:

.. code-block :: sh

    spool --load=./20171107_20171201.geo.unw.tif


Manual scene setup
------------------

Initialisation of a custom scene from python is also possible. Here we will import arbitrary data and define the reference frame manually.

.. code-block :: python
    :caption: Any 2D displacement data can be loaded into Kite, also pixel offsets!

    from kite import Scene

    sc = Scene()
    sc.displacement = num.empty((2048, 2048))
    
    # dummy line-of-sight vectors in radians
    sc.theta = num.full((2048, 2048), fill=num.pi/2)
    sc.phi = num.full((2048, 2048), fill=num.pi/4)


    # Reference the scene's frame
    sc.frame.llLat = 38.2095  # Lower-left corner latitude
    sc.frame.llLon = 19.1256  # Lower-left corner longitude
    sc.frame.dLat = .00005  # Latitudal pixel spacing in degree
    sc.frame.dLon = .00012  # Longitudal pixel spacing in degree



Inspect an InSAR scene with ``spool`` GUI
------------------------------------------

You can use start :doc:`../tools/spool` to inspect the scene and manipulate it's properties.

.. code-block :: python
    :caption: Kite's GUI ``spool`` is based on `Qt5 <https://www.qt.io/>`_. Here we will import data, straight from a GMT5SAR scene.

    from kite import Scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.spool()

Alternatively ``spool`` can be started from command line

.. code-block :: sh

    # Start spool and import a displacement scene data
    spool --load unwrap_ll.grd

    # Or load from Kite format
    spool my_scene.yml


Save scene and properties
-------------------------

The native file structure of ``Kite`` is based on NumPy binary files together with `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration files which hold the all information to and configurable parameters:

* :class:`~kite.Quadtree`,
* :class:`~kite.Covariance`,
* and :class:`~kite.scene.Meta`.

Also the expensive calculation of :attr:`kite.Covariance.covariance_matrix` is saved and preserved in the YAML file!

This code snippet shows how to import data from a foreign file format and transferring it to kite's native format.

.. code-block :: python
    :caption: Import data and save it in Kite format.

    from kite import Scene

    # The .grd is interpreted as an GMT5SAR scene
    sc = Scene.import_data('unwrap_ll.grd')

    # Writes out the scene in kite's native format
    sc.save('kite_scene')


Kite's file structure consists of only two files:

.. code-block :: sh

    kite_scene.yml
    kite_scene.npz
