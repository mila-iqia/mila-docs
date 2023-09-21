Data
====


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/data>`_


**job.sh**

.. literalinclude:: job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: main.py
   :language: python


**data.py**

.. literalinclude:: data.py
   :language: python


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
