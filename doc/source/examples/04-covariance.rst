Covariance Calculation
=========================================

This is a brief document describing where we learn how to access how we can calculate and use the covariance and weight properties of the quadtree. More, detailed information how  ``kite`` calculates the covariance matrix can be found at the module docmentation :mod:`kite.Covariance`.


Accessing the Covariance
------------------------

In this example we will access the covariance attributes of a pre-configured scene.

.. literalinclude :: /examples/scripts/covariance-calculation.py
    :language: python


Graphical Inspection and Tuning of the Covariance
--------------------------------------------------------
The covariance can be manipulated through ``spool``. More information here :ref:`spool-covariance`.

Manual Covariance Tuning
------------------------
The covariance can be manipulated by editing the ``.yml`` file. Or by changing the covariance parameters during runtime:

.. code-block:: python
    
    scene.covariance.a = 1.412e-4
    scene.covariance.b = 1.2521

    # Will show the updated covariance
    plt.imshow(scene.covariance.covariance_focal)

    # Will calculate the covariance matrix
    scene.covariance
    scene.save('filename')
