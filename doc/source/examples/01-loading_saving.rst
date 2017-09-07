Importing, Loading and Saving
=============================

Import unwrapped displacement scene from different InSAR processors:

* `GMT5SAR <http://gmt.soest.hawaii.edu/projects/gmt5sar>`_
* `GAMMA Software <http://www.gamma-rs.ch/no_cache/software.html>`_
* `ROI_PAC <http://roipac.org/cgi-bin/moin.cgi>`_
* `ISCE <https://winsar.unavco.org/isce.html>`_


.. note ::

    See :mod:`kite.scene_io` for required file structures and definitions of respective formats.


Importing a displacement scene
------------------------------

We will start by importing a scene which has been processed by GMT5SAR.

::

    from kite import Scene

    # We import a unwrapped interferrogram scene.
    # The format shall be detected automatically
    # in this case processed a GMTSAR5

    sc = Scene.import_data('unwrap_ll.grd')
    sc.spool()


Scene set-up from scratch
---------------------------

Initialisation of a generic binary dataset is also possible. Here the reference frame has to be initialized manually.

::

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



Scene Inspection with ``spool``
--------------------------------


You can use start :doc:`../spool` to inspect the scene and manipulate it's properties.

::

    from kite import Scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.spool()

Alternatively ``spool`` can be started from command line


.. code-block :: sh

    # Start spool and import a displacement scene from a foreign format
    spool --load unwrap_ll.grd

    # Or load data from a native container
    spool my_scene.yml


Saving Scenes and Properties
-----------------------------

The native file structure of ``kite`` is based on NumPy binary files together with `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration files holding the necessary information to fully reconstruct instances of

* :class:`~kite.Quadtree`,
* :class:`~kite.Covariance`,
* and :class:`~kite.scene.Meta`,

through serializing the correspondig *Config* classes (:doc:`../reference/index`).

.. note ::
 The expensive calculation of :attr:`kite.Covariance.covariance_matrix` is saved in the YAML file!


Importing data from a foreign file format and transfering it to kite's native format is as easy as 1, 2, 3...

::

    from kite import Scene

    # The .grd is interpreted as an GMT5SAR scene
    sc = Scene.import_data('unwrap_ll.grd')

    # Writes out the scene in kite's native format
    sc.save('kite_scene')


The kite file structure consists of only two files:

:: sh

    kite_scene.yml
    kite_scene.npz
