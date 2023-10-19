.. _many_tasks_per_gpu:

Launch many tasks on same GPU
=============================


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/many_tasks_per_gpu>`_

**job.sh**

.. literalinclude:: job.sh
    :language: bash


**main.py**

.. literalinclude:: main.py
    :language: python


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
