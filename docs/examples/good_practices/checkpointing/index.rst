Checkpointing
=============


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/checkpointing>`_


**job.sh**

.. literalinclude:: job.sh.diff
   :language: diff

**pyproject.toml**

.. literalinclude:: pyproject.toml
   :language: toml

**main.py**

.. literalinclude:: main.py.diff
   :language: diff


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
