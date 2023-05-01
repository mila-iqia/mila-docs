Checkpointing
=============


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/data/checkpointing>`_


**job.sh**

.. literalinclude:: examples/data/checkpointing/job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: examples/data/checkpointing/main.py.diff
   :language: diff


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
