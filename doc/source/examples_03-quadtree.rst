Quadtree parametrisation
========================

The :class:`~kite.Quadtree` subsamples the InSAR displacement in order to have a reduced and thus more manageable dataset for displacement modelling. The tree consists of :class:`~kite.quadtree.QuadNode` s, a state or slice of the tree is denoted as :attr:`~kite.Quadtree.leafs`.

Four parameters characterize the quadtree and when a :class:`~kite.quadtree.QuadNode` is split:

* epsilon/std threshold (:attr:`~kite.Quadtree.epsilon`)
* Fraction of NaN values within (:attr:`~kite.Quadtree.nan_allowed`)
* Maximum and minium dimension of the tile
  (:attr:`~kite.Quadtree.tile_size_max` and :attr:`~kite.Quadtree.tile_size_min`)


.. note :: All nodes of the :class:`~kite.Quadtree` are built upon initialisation an instance.

The parametrization of the tree can be done through the :doc:`spool` or the python interface of :class:`~kite.Quadtree`.

::
    
    from sc import Scene
    sc = Scene.import_data('test/data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

    sc.quadtree  # this initializes and constructs the tree

    sc.epsilon = .024
    sc.nan_allowed = .9
    sc.tile_size_max = 12000
    sc.tile_size_min = 250

    print(sc.reduction_rms)  # In units of [m]
    >>> 0.234123152

    for l in sc.quadtree.leafs:
        print l

    # One may export the tree in a CSV format
    sc.quadtree.export('/tmp/tree.csv')


.. rubric:: Footnotes

.. [#f1]  Jónsson, Sigurjón, Howard Zebker, Paul Segall, and Falk Amelung. 2002. “Fault Slip Distribution of the 1999 Mw 7.1 Hector Mine, California, Earthquake, Estimated from Satellite Radar and GPS Measurements.” Bulletin of the Seismological Society of America 92 (4): 1377–89. doi:10.1785/0120000922.


