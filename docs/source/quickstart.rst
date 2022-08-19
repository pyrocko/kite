Quickstart
==========

Kite supports importing unwrapped displacement scenes from different InSAR processors:

* `GMT5SAR <https://gmt.soest.hawaii.edu/projects/gmt5sar>`_
* `GAMMA Software <http://www.gamma-rs.ch/no_cache/software.html>`_
* `ROI_PAC <http://www.geo.cornell.edu/eas/PeoplePlaces/Faculty/matt/roi_pac.html/>`_
* `ISCE <https://winsar.unavco.org/software/isce>`_
* `SARscape <http://www.sarmap.ch/page.php?page=sarscape>`_
* `LiCSAR <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_
* `ARIA  <https://aria.jpl.nasa.gov/products>`_
* `SNAP <https://step.esa.int/main/toolboxes/snap/>`_

Each processor delivers different file formats and metadata. In order to import the data into Kite, data has to be prepared. Details for each format are described in :mod:`kite.scene_io`.


Importing data to ``spool``
---------------------------

When you have processed your InSAR data and aligned with the expected import conversions (:mod:`kite.scene_io`), you can use spool to import the data:

.. code-block :: sh
    :caption: Importing and visualising unwrapped InSAR data with ``spool``.

    spool --load data/my_scene-asc.grd


Inspecting a scene with ``spool``
---------------------------------

You can use start :doc:`../tools/spool` to inspect the scene and manipulate it's properties.

.. code-block :: python
    :caption: Kite's GUI ``spool`` is based on `Qt5 <https://www.qt.io/>`_. Here we will import data, straight from a GMT5SAR scene.

    from kite import Scene
    sc = Scene.import_file('unwrap_ll.grd')

    # Start the GUI
    sc.spool()

.. code-block :: sh
    :caption: Alternatively ``spool`` can be started from command line.

    # Start spool and import a displacement scene data
    spool --load unwrap_ll.grd

    # Or load from Kite format
    spool my_scene.yml


.. figure :: ../_images/spool-scene.png
    :width: 90%
    :align: center

    **Figure 1**: Manipulating unwrapped InSAR surface displacement data in Spool, Kite's GUI.


Scripted data import
--------------------

We will start with importing a scene from GMT5SAR using a Python script:

.. code-block :: python
    :caption: GMT5SAR is an open-source processor based on GMT. We will import a binary ``.grd`` file.

    from kite import Scene

    # We import a unwrapped interferogram scene.
    # The format shall be detected automatically
    # in this case processed a GMTSAR5

    sc = Scene.import_data('unwrap_ll.grd')

    # Open spool
    sc.spool()


Scripted scene setup
--------------------

Initialisation of a custom scene through a Python script. Here we will import arbitrary data and define the geographical reference frame manually.

.. code-block :: python
    :caption: Setting up scene from 2D displacement data.

    from kite import Scene

    sc = Scene()
    sc.displacement = num.empty((2048, 2048))

    # Dummy line-of-sight vectors in radians
    # Theta is the elevation angle towards satellite from horizon in radians.
    sc.theta = num.full((2048, 2048), fill=num.pi/2)
    # Phi, the horizontal angle towards satellite in radians, counter-clockwise from East.
    sc.phi = num.full((2048, 2048), fill=num.pi/4)


    # Reference the scene's frame lower left corner, always in geographical coordinates
    sc.frame.llLat = 38.2095
    sc.frame.llLon = 19.1256

    # The pixel spacing can be either 'meter' or 'degree'
    sc.frame.spacing = 'degree'
    sc.frame.dN = .00005  # Latitudal pixel spacing
    sc.frame.dE = .00012  # Longitudal pixel spacing

    # Saving the scene
    sc.save('my_scene')


Saving the scene and quadtree/covariance
----------------------------------------

The native file structure of ``Kite`` is based on NumPy binary files together with `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration files, holding all meta information and configurable parameters, such as:

* :class:`~kite.Quadtree`,
* :class:`~kite.Covariance`,
* and :class:`~kite.scene.Meta`.

This structure also holds the :attr:`kite.Covariance.covariance_matrix`, which requires a computational intensive task!

This code snippet shows how to import data from a foreign file format and saving it to kite's native format.

.. code-block :: python
    :caption: Importing data and saving it in Kite format.

    from kite import Scene

    # The .grd is interpreted as an GMT5SAR scene
    sc = Scene.import_data('unwrap_ll.grd')

    # Writes out the scene in kite's native format
    sc.save('kite_scene')



.. code-block :: sh
    :caption: Kite's file structure consists of only two files:

    kite_scene.npz
    kite_scene.yml


Download and import data from ARIA (NASA)
-----------------------------------------

The `ARIA <https://aria.jpl.nasa.gov/>`_ web service provides unwrapped Sentinel-1 data covering selected regions. The data can be explore on the `website <https://aria-products.jpl.nasa.gov/>`_. For this example we use ``wget`` to download the ascending and descending ARIA GUNW data products from ARIA from the `2019 Ridgecrest Earthquakes <https://en.wikipedia.org/wiki/2019_Ridgecrest_earthquakes>`_.

.. code-block :: sh

    # Ascending
    wget https://aria-products.jpl.nasa.gov/search/dataset/grq_v2.0.2_s1-gunw-released/S1-GUNW-A-R-064-tops-20190710_20180703-015013-36885N_35006N-PP-9955-v2_0_2/S1-GUNW-A-R-064-tops-20190710_20180703-015013-36885N_35006N-PP-9955-v2_0_2.nc

    # Descending
    wget https://aria-products.jpl.nasa.gov/search/dataset/grq_v2.0.2_s1-gunw-released/S1-GUNW-D-R-071-tops-20190728_20190622-135213-36450N_34472N-PP-b4b2-v2_0_2/S1-GUNW-D-R-071-tops-20190728_20190622-135213-36450N_34472N-PP-b4b2-v2_0_2.nc


Now use the Python package `ARIA-tools <https://github.com/aria-tools/ARIA-tools>`_ to extract three channels (``unwrappedPhase, incidenceAngle, azimuthAngle``) from the ARIA data product:

.. code-block :: sh
    :caption: We use two workdirs: ``ascending`` and ``descending``.

    # Extract into ascending/
    ariaExtract.py -w ascending -f S1-GUNW-A-R-064-tops-20190710_20180703-015013-36885N_35006N-PP-9955-v2_0_2.nc -d download -l unwrappedPhase,incidenceAngle,azimuthAngle

    # Extract into descending/
    ariaExtract.py -w descending -f S1-GUNW-D-R-071-tops-20190728_20190622-135213-36450N_34472N-PP-b4b2-v2_0_2.nc  -d download -l unwrappedPhase,incidenceAngle,azimuthAngle

For more information see ``ariaExtract.py --help`` and the `ARIA-tools documentation <https://github.com/aria-tools/ARIA-tools>`_.

Import ARIA GUNW data into Kite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To import the data into spool use a regular ``load``:

.. code-block :: sh

    spool --load ascending


.. note ::

    ARIA scenes are provided at a high resolution and calculation of the covariance matrix can take a long time!



Download and import data from COMET LiCSAR
------------------------------------------

A slim client for downloading `COMET LiCSAR <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_ products is included in :mod:`kite.clients`. The script will download the passed unwrapped LiCSAR data and necessary LOS GeoTIFFs into the current directory.

This example will download data from the 2017 Iran–Iraq earthquake (M 7.3) from the `COMET LiCSAR Portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_:


.. code-block :: sh
    :caption: Download data from COMET LiCSAR Portal.

    python3 -m kite.clients http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/6/006D_05509_131313/products/20171107_20171201/20171107_20171201.geo.unw.tif .

Now just load the scene into Kite or spool.

.. code-block :: sh
    :caption: Importing the scene in spool.

    spool --load 20171107_20171201.geo.unw.tif


Converting StaMPS velocities to a Kite scene
--------------------------------------------

The CLI tool :file:`stamps2kite` loads PS velocities from a `StaMPS <https://homepages.see.leeds.ac.uk/~earahoo/stamps/>`_ project (i.e. processed mean velocity data through ``ps_plot(..., -1);``), and grids the data into mean velocity bins. The LOS velocities will be converted into a Kite scene.

StaMPS' data has to be fully processed through and may stem from the master
project or from one of the processed small baseline pairs. The required files are:

- :file:`ps2.mat`          Meta information and geographical coordinates.
- :file:`parms.mat`        Meta information about the scene (heading, etc.).
- :file:`ps_plot*.mat`     Processed and corrected LOS velocities.
- :file:`mv2.mat`          Mean velocity's standard deviation.

- :file:`look_angle.1.in`  Look angles for the scene.
- :file:`width.txt`        Width dimensions of the interferogram and
- :file:`len.txt`          length.


.. code-block :: sh
    :caption: Importing StaMPS mean velocities into a gridded Kite scene.

    stamps2kite stamps_project/ --resolution 800 800 --save my_kite_scene


For more information on the util, see the ``--help``.
