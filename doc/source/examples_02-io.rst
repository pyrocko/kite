Saving and loading Scenes
=========================

The native file structure of ``kite`` is based on NumPy binary files together with `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration files holding the necessary information to fully reconstruct instances of

* :class:`~kite.Quadtree`,
* :class:`~kite.Covariance`,
* and :class:`~kite.scene.Meta`,

through serializing the correspondig *Config* classes (:doc:`modules`).

.. note ::
 The expensive calculation of :attr:`kite.Covariance.covariance_matrix` is saved in the YAML file!

Importing data from a foreign file format and transfering it to kite's native format is as easy as 1, 2, 3...

::

    from kite import Scene
    sc = Scene.import_data('unwrap_ll.grd')  # GMT5SAR import
    sc.save('kite_scene')

The kite file structure consists of only two files:

::

    project_name.yml
    project_name.npz
