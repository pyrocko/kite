Quadtree Parametrisation
========================

The :class:`~kite.Quadtree` subsamples the InSAR displacement in order to have a reduced and thus more manageable dataset for displacement modelling. The tree consists of :class:`~kite.quadtree.QuadNode` s, a state or slice of the tree is denoted as :attr:`~kite.Quadtree.leaves`.

Four parameters characterize the quadtree and when a :class:`~kite.quadtree.QuadNode` is split:

* epsilon/variance threshold of the leaf/tile(:attr:`~kite.Quadtree.epsilon`)
* Fraction of NaN values within (:attr:`~kite.Quadtree.nan_allowed`)
* Maximum and minium dimension of the tile
  (:attr:`~kite.Quadtree.tile_size_max` and :attr:`~kite.Quadtree.tile_size_min`)


Basically the quadtree implements ideas from [#f1]_.

.. note :: All nodes of the :class:`~kite.Quadtree` are built upon initialisation an instance.


Graphical quadtree parametrisation
----------------------------------

The GUI ``spool`` offers an interactive parametrisation of the quadtree. Start the programm and play around on tab :guilabel:`&Quadtree`.


Scripted quadtree parametrisation
---------------------------------

The parametrization of the tree can be done through the :doc:`../spool` or the python interface of :class:`~kite.Quadtree`.

::
    
    from sc import Scene
    sc = Scene.import_data('test/data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

    qt = sc.quadtree  # For convenience we set an abbreviation to the quadtree

    # Parametrisation of the quadtree
    qt.epsilon = 0.024  # Variance threshold
    qt.nan_allowed = 0.9  # Percentage of NaN values allowed per tile/leaf
    qt.tile_size_max = 12000  # Maximum leaf size in [m]
    qt.tile_size_min = 250  # Minimum leaf size in [m]

    print(qt.reduction_rms)  # In units of [m]
    >>> 0.234123152

    for l in qt.leafs:
        print l

    # We save the quadtree configuration together w the scene
    sc.save('kite_scene')

    # Or one may export the tree in a CSV format
    qt.export('/tmp/tree.csv')


.. rubric:: Footnotes

.. [#f1]  Jónsson, Sigurjón, Howard Zebker, Paul Segall, and Falk Amelung. 2002. “Fault Slip Distribution of the 1999 Mw 7.1 Hector Mine, California, Earthquake, Estimated from Satellite Radar and GPS Measurements.” Bulletin of the Seismological Society of America 92 (4): 1377–89. doi:10.1785/0120000922.


