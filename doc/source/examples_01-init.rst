Scene import/initialization
===========================

Import unwrapped displacement scene from ``GMT5SAR``, ``GAMMA`` or ``ISCE``.
See module documentation for file structure definition

::

    from kite import Scene

    # We import a GMT5SAR scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.plot()


Manual initialisation of a generic binary dataset is also possible:

::

    from kite import Scene

    sc = Scene()
    sc.displacement = num.empty((2048, 2048))

    # Frame the scene
    sc.frame.llLat = 38.2095
    sc.frame.llLon = 19.1256
    sc.frame.dLat = .00005
    sc.frame.dLon = .00012


Start :doc:spool for scene inspection and manipulation

::

    sc.spool()