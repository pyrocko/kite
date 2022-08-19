Quadtree
========

The :class:`~kite.Quadtree` reduces the amount of displacement data by sub-sampling the InSAR displacement map. For efficient forward modelling it is important to have reasonably sized dataset.

.. figure :: ../../_images/quadtree-myanmar-quadtree.gif
    :width: 100%

    Figure 1: Quadtree data reduction of an InSAR displacement scene. (Left) the origin full resolution ALSO displacement map, (right) the adaptive quadtree reduction.

The quadtree is made from hierarchically organized :class:`~kite.quadtree.QuadNode`, a slice through of the tree's nodes is then called :attr:`~kite.Quadtree.leaves`.

Parameters defining the quadtree are:

* :attr:`~kite.Quadtree.epsilon` threshold controlling the leaves' split, this is the displacement variance within a leave.
* :attr:`~kite.Quadtree.nan_allowed` is the fraction of allowed NaN values before a leaf is dismissed.
* :attr:`~kite.Quadtree.tile_size_max` and :attr:`~kite.Quadtree.tile_size_min` define the maximum and minimum dimension of the tile in meter or degree

Kite realises the quadtree concept from Jónsson et al. (2002) [#f1]_.

.. note :: All nodes of the :class:`~kite.Quadtree` are built upon initialisation an instance.


Interactive quadtree parametrisation
------------------------------------

The graphical user interface (GUI) ``spool`` offers an interactive parametrisation of the quadtree. Start the program, click on tab :guilabel:`&Quadtree`. Detailed instruction can be found in :doc:`spool's tutorial </tools/spool>`.

.. code-block :: sh
    :caption: Start spool and open a QuadTree container.

    spool insar_displacement_scene.npz


Scripted quadtree parametrisation
---------------------------------

The quadtree can also be parametrised by a python script. This example modifies the quadtree and saves the scene.

.. tip ::

    It is recommended to use the spool GUI for parametrisation of the quadtree and covariance.

.. code-block :: python
    :caption: Scripted parametrisation of the quadtree.

    import logging
    from kite import Scene

    logging.basicConfig(level=logging.DEBUG)

    sc = Scene.import_data('test/data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat')

    # For convenience we set an abbreviation to the quadtree
    qt = sc.quadtree

    # Parametrisation of the quadtree
    qt.epsilon = 0.024        # Variance threshold
    qt.nan_allowed = 0.9      # Percentage of NaN values allowed per tile/leave

    # Be careful here, if you scene is referenced in degree use decimal values!
    qt.tile_size_max = 12000  # Maximum leave edge length in [m] or [deg]
    qt.tile_size_min = 250    # Minimum leave edge length in [m] or [deg]

    print(qt.reduction_rms)   # In units of [m] or [deg]
    # >>> 0.234123152

    for l in qt.leaves:
        print l

    # We save the scene in kite's format
    sc.save('kite_scene')

    # Or export the quadtree to CSV file
    qt.export('/tmp/tree.csv')


Plotting the quadtree with matplotlib
-------------------------------------

We can also use Matplotlib to plot the Quadtree's current leaves.

.. code-block :: python
    :caption: Plotting the quadtree with Matplotlib.

    import numpy as num

    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    from kite import Scene

    sc = Scene.load('my_scene.yml')
    qt = sc.quadtree

    fig = plt.figure()
    ax = fig.gca()

    limit = num.abs(qt.leaf_medians).max()
    color_map = cm.ScalarMappable(
        norm=colors.Normalize(vmin=-limit, vmax=limit),
        cmap=cm.get_cmap('RdBu'))

    for rect, leaf in zip(qt.getMPLRectangles(), qt.leaves):
        color = color_map.to_rgba(leaf.median)
        rect.set_facecolor(color)
        ax.add_artist(rect)

    ax.set_xlim(qt.leaf_eastings.min(), qt.leaf_eastings.max())
    ax.set_ylim(qt.leaf_northings.min(), qt.leaf_northings.max())

    plt.show()


.. rubric:: Footnotes

.. [#f1]  Jónsson, Sigurjón, Howard Zebker, Paul Segall, and Falk Amelung. 2002. “Fault Slip Distribution of the 1999 Mw 7.1 Hector Mine, California, Earthquake, Estimated from Satellite Radar and GPS Measurements.” Bulletin of the Seismological Society of America 92 (4): 1377–89. doi:10.1785/0120000922.
