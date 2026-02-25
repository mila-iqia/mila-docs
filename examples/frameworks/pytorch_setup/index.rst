.. _pytorch_setup:

PyTorch Setup
=============

**Prerequisites**: (Make sure to read the following before using this example!)


The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_


* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`uv`


**job.sh**


.. literalinclude:: job.sh
    :language: bash


**pyproject.toml**

.. literalinclude:: pyproject.toml
    :language: toml


**main.py**

.. literalinclude:: main.py
    :language: python


**Running this example**

This assumes that you already installed UV on the cluster you are working on.

To create this environment, we first request resources for an interactive job.
Note that we are requesting a GPU for this job, even though we're only going to
install packages. This is because we want PyTorch to be installed with GPU
support, and to have all the required libraries.

.. code-block:: bash

    # On the Mila cluster: (on DRAC/PAICE, run `uv sync` on a login node)
    $ salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00
    salloc: --------------------------------------------------------------------------------------------------
    salloc: # Using default long partition
    salloc: --------------------------------------------------------------------------------------------------
    salloc: Pending job allocation 2959785
    salloc: job 2959785 queued and waiting for resources
    salloc: job 2959785 has been allocated resources
    salloc: Granted job allocation 2959785
    salloc: Waiting for resource configuration
    salloc: Nodes cn-g022 are ready for job
    $ # Create the virtual environment and install all dependencies
    $ uv sync
    (...)
    $ # Optional: Activate the environment and run the python script:
    $ . .venv/bin/activate
    $ python main.py

You can exit the interactive job once the environment has been created.
Then, you can submit a job to run the example with sbatch:

.. code-block:: bash

    $ sbatch job.sh
