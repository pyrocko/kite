The Sources Module
=======================

.. autoclass::
    kite.sources
    :members:

The Okada Model
---------------------------

Surface displacement from Okada [#f1]_ sources are calculated using a wrapped and parallelized (OpenMP)``disloc`` code.

.. [#f1] Okada, Y., 1992, Internal deformation due to shear and tensile faults in a half-space, Bull. Seism. Soc. Am., 82, 1018-1040.


.. autoclass::
    kite.sources.OkadaSource
    :members:
    :show-inheritance:


Pyrocko Source Models
---------------------

Pyrocko source models are calculated using :mod:`pyrocko.gf`.

.. autoclass::
    kite.sources.PyrockoSource
    :members:
    :show-inheritance:


.. autoclass::
    kite.sources.PyrockoRectangularSource
    :members:
    :show-inheritance:

.. autoclass::
    kite.sources.PyrockoMomentTensor
    :members:
    :show-inheritance:


.. autoclass::
    kite.sources.PyrockoDoubleCouple
    :members:
    :show-inheritance:

.. autoclass::
    kite.sources.PyrockoRingfaultSource
    :members:
    :show-inheritance:


Compound Dislocation Models
----------------------------

Compound dislocation models developed by Mehdi Nikkhoo [#f2]_. Are implemented in Python.
More information on the routines and sources at <http://www.volcanodeformation.com>_.

.. [#f2] Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017): Compound dislocation models (CDMs) for volcano deformation analyses. - Geophysical Journal International, 208, 2, p. 877-894.

.. autoclass::
    kite.sources.EllipsoidSource
    :members:
    :show-inheritance:

.. autoclass::
    kite.sources.PointCompoundSource
    :members:
    :show-inheritance:

Objects Bases
-------------

.. autoclass::
    kite.sources.base.SandboxSource
    :members:

.. autoclass::
    kite.sources.base.SandboxSourceRectangular
    :members:
    :show-inheritance:
