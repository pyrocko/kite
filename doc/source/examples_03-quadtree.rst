Quadtree parametrisation
========================

The :class:`~kite.Quadtree` subsamples the InSAR displacement in order to have a reduced and thus more manageable dataset for displacement modelling. The tree consists of :class:`~kite.quadtree.QuadNode` s, a state or slice of the tree is denoted as :attr:`~kite.Quadtree.leafs`.

Four parameters characterize the quadtree and when a :class:`~kite.quadtree.QuadNode` is split:

* epsilon/std threshold (:attr:`~kite.Quadtree.epsilon`)
* Fraction of NaN values within (:attr:`~kite.Quadtree.nan_allowed`)
* Maximum and minium dimension of the tile
  (:attr:`~kite.Quadtree.tile_size_max` and :attr:`~kite.Quadtree.tile_size_min`)


.. note :: The whole :class:`~kite.Quadtree` and is built upon initialisation an instance, 

.. rubric:: Footnotes

.. [#f1]  Jónsson, Sigurjón, Howard Zebker, Paul Segall, and Falk Amelung. 2002. “Fault Slip Distribution of the 1999 Mw 7.1 Hector Mine, California, Earthquake, Estimated from Satellite Radar and GPS Measurements.” Bulletin of the Seismological Society of America 92 (4): 1377–89. doi:10.1785/0120000922.


