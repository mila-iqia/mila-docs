.. _jax_setup:

Jax Setup
=========

**Prerequisites**: (Make sure to read the following before using this example!)

* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`Conda`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax_setup>`_


**job.sh**

.. literalinclude:: job.sh.diff
   :language: diff


**main.py**

.. literalinclude:: main.py
   :language: python


**Running this example**

This assumes that you already created a conda environment named "jax". To create
this environment, we first request resources for an interactive job.  Note that
we are requesting a GPU for this job, even though we're only going to install
packages. This is because we want Jax to be installed with GPU support, and to
have all the required libraries.

Jax comes with precompiled binaries targetting a specific version of CUDA. In
case you encounter an error like the following:

.. code-block::

   The NVIDIA driver's CUDA version is 11.7 which is older than the ptxas CUDA
   version (11.8.89). Because the driver is older than the ptxas version, XLA is
   disabling parallel compilation, which may slow down compilation. You should
   update your NVIDIA driver or use the NVIDIA-provided CUDA forward
   compatibility packages.

Try installing the specified version of CUDA in conda :
https://anaconda.org/nvidia/cuda. E.g. ``"nvidia/label/cuda-11.8.0"`` if ptxas
CUDA version is 11.8.XX

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
    $ # Load anaconda
    $ module load anaconda/3
    $ # Create the environment (see the example):
    $ conda create -y -n jax_ex -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
    (...)
    $ # Press 'y' to accept if everything looks good.
    (...)
    $ # Activate the environment:
    $ conda activate jax_ex
    # Install Jax using `pip`
    # *Please note* that as soon as you install packages from `pip install`, you
    # should not install any more packages using `conda install`
    $ pip install --upgrade "jax[cuda11_pip]" \
    $    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Exit the interactive job once the environment has been created. Then, the
example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh
