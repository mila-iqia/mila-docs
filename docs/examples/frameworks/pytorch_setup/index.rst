.. _pytorch_setup:

PyTorch Setup
=============

**Prerequisites**: (Make sure to read the following before using this example!)


The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_


* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`Conda`


**job.sh**


.. literalinclude:: job.sh
    :language: bash


**main.py**


.. literalinclude:: main.py
    :language: python


**Running this example**

This assumes that you already created a conda environment named "pytorch". To
create this environment, we first request resources for an interactive job.
Note that we are requesting a GPU for this job, even though we're only going to
install packages. This is because we want PyTorch to be installed with GPU
support, and to have all the required libraries.

.. literalinclude:: examples/frameworks/pytorch_setup/make_env.sh
    :language: bash

Exit the interactive job once the environment has been created. Then, the
example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh
