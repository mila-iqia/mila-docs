Hugging Face Dataset
====================


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/data/hf>`_


**job.sh**

.. literalinclude:: examples/data/hf/job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: examples/data/hf/main.py.diff
   :language: diff


**prepare_data.py**

.. literalinclude:: examples/data/hf/prepare_data.py
   :language: python


**data.sh**

.. literalinclude:: examples/data/hf/data.sh
   :language: bash


**get_dataset_cache_files.py**

.. literalinclude:: examples/data/hf/get_dataset_cache_files.py
   :language: python


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
