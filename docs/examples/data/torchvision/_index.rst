Torchvision
===========


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/data/torchvision>`_


**job.sh**

.. literalinclude:: examples/data/torchvision/job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: examples/data/torchvision/main.py.diff
   :language: diff


**data.py**

.. literalinclude:: examples/data/torchvision/data.py
   :language: python


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
