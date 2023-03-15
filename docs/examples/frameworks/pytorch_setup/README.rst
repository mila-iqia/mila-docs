PyTorch Setup
===================

.. IDEA: Add a link to all the sections of the documentation that have to
.. absolutely have been read before this tutorial.

**Prerequisites**: (Make sure to read the following before using this example!)

* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`Conda`


**job.sh**


.. literalinclude:: /examples/frameworks/pytorch_setup/job.sh
    :language: bash


**main.py**


.. literalinclude:: /examples/frameworks/pytorch_setup/main.py
    :language: python


**Running this example**

This assumes that you already created a conda environment named "pytorch".
To create this environment, we first request resources for an interactive job.
Note that we are requesting a GPU for this job, even though we're only going to install packages.
This is because we want PyTorch to be installed with GPU support, and to have all the required
libraries.

.. code-block:: bash

    $ salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00
    salloc: --------------------------------------------------------------------------------------------------
    salloc: # Using default long partition
    salloc: --------------------------------------------------------------------------------------------------
    salloc: Pending job allocation 2959785
    salloc: job 2959785 queued and waiting for resources
    salloc: job 2959785 has been allocated resources
    salloc: Granted job allocation 2959785
    salloc: Waiting for resource configuration
    salloc: Nodes cn-g022 are ready for job
    $ # Create the environment (see the example):
    $ conda create -n pytorch python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    (...)
    $ # Press 'y' to accept if everything looks good.
    (...)
    $ # Activate the environment:
    $ conda activate pytorch

Exit the interactive job once the environment has been created.
Then, the example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh