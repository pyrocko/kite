Modelling static displacement
==============================

Kite comes with a :class:`~kite.SandboxScene`, which is a playground for static displacement sources of different kinds and modelling engines. It support analytical backends such as Okada ``disloc``-code [#f1]_ and Compound dislocation model ([#f2]_; http://www.volcanoedeformation.com/). Numerical forward modelling is enabled through :mod:`pyrocko.gf`, this allows us to put geometrically more complex sources into the modelling sandbox.

Currently implemented static displacement sources:

* :class:`~kite.sources.OkadaSource`

Several Pyrocko Sources:

* :class:`~kite.sources.PyrockoRectangularSource`
* :class:`~kite.sources.PyrockoMomentTensor`
* :class:`~kite.sources.PyrockoDoubleCouple`
* :class:`~kite.sources.PyrockoRingfaultSource`

Analytical dilatational point through CDM ([#f2]_) sources:

* :class:`~kite.sources.EllipsoidSource`
* :class:`~kite.sources.PointCompoundSource`

More information about the sources and their implementation can be found at the modules reference: :mod:`kite.sources`.

.. [#f1] Okada, Y., 1992, Internal deformation due to shear and tensile faults in a half-space, Bull. Seism. Soc. Am., 82, 1018-1040.

.. [#f2] Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017): Compound dislocation models (CDMs) for volcano deformation analyses. - Geophysical Journal International, 208, 2, p. 877-894.


Add Displacement Sources into :class:`SandboxScene`
---------------------------------------------------

In this example we will add a simple :class:`~kite.sources.OkadaSource` into a :class:`~kite.SandboxScene`.

.. literalinclude :: /examples/scripts/sandboxscene-add_source.py
    :language: python

A full list of available sources and their parameters can be found at the modules' reference page :mod:`kite.sources`.

Save and Load :class:`pyrocko.SandboxScene`
--------------------------------------------

In this small example we will add a basic :class:`~kite.sources.EllipsoidSource` to the sandbox. Subsequently we will save it and load it again.

.. literalinclude :: /examples/scripts/sandboxscene-save_load.py
    :language: python


Graphical Manipulation of Displacement Sources
-----------------------------------------------

The graphical user interface :ref:`talpa` offers tools to handle and interact with the different kinds of displacement sources.
