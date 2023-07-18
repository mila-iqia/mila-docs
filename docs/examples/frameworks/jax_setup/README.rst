.. NOTE: This file is auto-generated from examples/frameworks/jax_setup/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

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

.. code:: diff

    # frameworks/pytorch_setup/job.sh -> frameworks/jax_setup/job.sh
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=16G
    #SBATCH --time=00:15:00

    set -e  # exit on error.
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"

    # Ensure only anaconda/3 module loaded.
    module --quiet purge
    # This example uses Conda to manage package dependencies.
    # See https://docs.mila.quebec/Userguide.html#conda for more information.
    module load anaconda/3

    # Creating the environment for the first time:
   -# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
   -#     pytorch-cuda=11.6 -c pytorch -c nvidia
   -# Other conda packages:
   -# conda install -y -n pytorch -c conda-forge rich
   +# conda create -y -n jax -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
   +# conda activate jax
   +# Install Jax using `pip`
   +# *Please note* that as soon as you install packages from `pip install`, you
   +# should not install any more packages using `conda install`
   +# pip install --upgrade "jax[cuda11_pip]" \
   +#    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # Activate the environment:
   -conda activate pytorch
   +conda activate ~/CODE/mila-docs/.tmp/env/cp39/exjax_/

   -# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
   +
   +# Fixes issues with MIG-ed GPUs
    unset CUDA_VISIBLE_DEVICES

    python main.py


**main.py**

.. code:: diff

    # frameworks/pytorch_setup/main.py -> frameworks/jax_setup/main.py
   -import torch
   -import torch.backends.cuda
   +import jax
   +from jax.lib import xla_bridge


    def main():
   -    cuda_built = torch.backends.cuda.is_built()
   -    cuda_avail = torch.cuda.is_available()
   -    device_count = torch.cuda.device_count()
   +    device_count = len(jax.local_devices(backend="gpu"))
   +    print(f"Jax default backend:         {xla_bridge.get_backend().platform}")
   +    print(f"Jax-detected #GPUs:          {device_count}")

   -    print(f"PyTorch built with CUDA:         {cuda_built}")
   -    print(f"PyTorch detects CUDA available:  {cuda_avail}")
   -    print(f"PyTorch-detected #GPUs:          {device_count}")
        if device_count == 0:
            print("    No GPU detected, not printing devices' names.")
        else:
            for i in range(device_count):
   -            print(f"    GPU {i}:      {torch.cuda.get_device_name(i)}")
   +            print(f"    GPU {i}:      {jax.local_devices(backend='gpu')[0].device_kind}")


    if __name__ == "__main__":
        main()


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
    $ # Create the environment (see the example):
    $ conda create -y -n jax -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
    (...)
    $ # Press 'y' to accept if everything looks good.
    (...)
    $ # Activate the environment:
    $ conda activate jax
    # Install Jax using `pip`
    # *Please note* that as soon as you install packages from `pip install`, you
    # should not install any more packages using `conda install`
    $ pip install --upgrade "jax[cuda11_pip]" \
    $    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Exit the interactive job once the environment has been created. Then, the
example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh
