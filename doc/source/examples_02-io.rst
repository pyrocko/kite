Saving and loading Scenes
=========================

The native file structure of ``kite`` is based on binary files together with `YAML<https://en.wikipedia.org/wiki/YAML>` configuration files holding the necessary :py:class:`kite.Quadtree`, :py:class`kite.covariance` and :py:class:`kite.scene.Meta` information.

::

    from kite import Scene
    sc = Scene.import_file('unwrap_ll.grd')
    sc.save('kite_scene')

The fundamental file structure is:

::

    project_name.yml
    project_name.disp.npy
    project_name.phi.npy
    project_name.theta.npy
