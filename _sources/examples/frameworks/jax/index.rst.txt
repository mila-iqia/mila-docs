.. _jax:

Jax
===


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/jax_setup/index`
* :doc:`/examples/distributed/single_gpu/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax>`_


**job.sh**

.. literalinclude:: job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: main.py.diff
   :language: diff


**model.py**

.. literalinclude:: model.py
   :language: python


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
