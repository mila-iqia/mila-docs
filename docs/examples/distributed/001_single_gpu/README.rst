001 - Single GPU Job
====================


**Prerequisites**
Make sure to read the following sections of the documentation before using this example:

* :ref:`pytorch_setup`

**job.sh**


.. literalinclude:: examples/distributed/001_single_gpu/job.sh
    :language: bash


**main.py**


.. literalinclude:: examples/distributed/001_single_gpu/main.py
    :language: python


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
