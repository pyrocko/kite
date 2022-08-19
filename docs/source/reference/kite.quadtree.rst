``Quadtree`` Module
===================

The quadtree reduces the amount of displacement information contained in a full resolution displacement scene by adaptively subsampling in areas of complex signal (a visual example can be found here :doc:`../tools/spool`. The complexity of the signal is defined through the standard deviation within a tile (or :class:`~kite.quadtree.QuadNode`) of the data. The quadtree parameters define thresholds when a :class:`~kite.quadtree.QuadNode` is split. The four essential threshold controlling the tree are:

* epsilon/std threshold (:attr:`~kite.Quadtree.epsilon`)
* Fraction of NaN values within (:attr:`~kite.Quadtree.nan_allowed`)
* Maximum and minimum dimension of the tile
  (:attr:`~kite.Quadtree.tile_size_max` and :attr:`~kite.Quadtree.tile_size_min`)

Programming example of the quadtree can be found here :doc:`../examples/quadtree` and here :doc:`../tools/spool`.

.. autoclass::
    kite.Quadtree
    :members:


QuadtreeConfig
------------------
The ``QuadtreeConfig`` holds the necessary configuration to reconstruct and save an instance.

.. autoclass:: kite.quadtree.QuadtreeConfig

QuadNode Object
------------------

.. autoclass:: kite.quadtree.QuadNode
    :members:
