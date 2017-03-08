Scene import/initialization
===========================

Import unwrapped displacement scene from `GMT5SAR <http://gmt.soest.hawaii.edu/projects/gmt5sar>`_, `GAMMA Software <http://www.gamma-rs.ch/no_cache/software.html>`_, `ROI_PAC <http://roipac.org/cgi-bin/moin.cgi>`_ or `ISCE <https://winsar.unavco.org/isce.html>`_.
See :module:`kite.scene_io` for required file structure definitions of respective formats.

::

    from kite import Scene

    # We import a GMT5SAR scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.plot()


Manual initialisation of a generic binary dataset is also possible. Here the fundamental frame has to be initialized manually.

::

    from kite import Scene

    sc = Scene()
    sc.displacement = num.empty((2048, 2048))
    
    # dummy line-of-sight vectors
    sc.theta = num.ones((2048, 2048))*192.
    sc.phi = num.ones((2048, 2048))*67.


    # Frame the scene
    sc.frame.llLat = 38.2095
    sc.frame.llLon = 19.1256
    sc.frame.dLat = .00005
    sc.frame.dLon = .00012


Start :doc:`spool` for scene inspection and manipulation

::

    from kite import Scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.spool()

Alternatively the ``spool`` can be started from commandline

.. code-block :: sh

    spool unwrap_ll.grd
