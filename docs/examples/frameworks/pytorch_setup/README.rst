.. NOTE: This file is auto-generated from examples/frameworks/pytorch_setup/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

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


.. code:: bash

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

   # Activate the environment:
   # NOTE: Use the `make_env.sh` script to create the environment if you haven't already.
   conda activate pytorch

   # Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
   unset CUDA_VISIBLE_DEVICES

   python main.py


**main.py**


.. code:: python

   import torch
   import torch.backends.cuda


   def main():
       cuda_built = torch.backends.cuda.is_built()
       cuda_avail = torch.cuda.is_available()
       device_count = torch.cuda.device_count()

       print(f"PyTorch built with CUDA:         {cuda_built}")
       print(f"PyTorch detects CUDA available:  {cuda_avail}")
       print(f"PyTorch-detected #GPUs:          {device_count}")
       if device_count == 0:
           print("    No GPU detected, not printing devices' names.")
       else:
           for i in range(device_count):
               print(f"    GPU {i}:      {torch.cuda.get_device_name(i)}")


   if __name__ == "__main__":
       main()


**Running this example**

This assumes that you already created a conda environment named "pytorch". To
create this environment, we first request resources for an interactive job.
Note that we are requesting a GPU for this job, even though we're only going to
install packages. This is because we want PyTorch to be installed with GPU
support, and to have all the required libraries.

.. code:: bash

   #!/bin/bash
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=1
   #SBATCH --mem=16G
   #SBATCH --time=00:30:00

   # NOTE: Run this either with `sbatch make_env.sh` or within an interactive job with `salloc`:
   # salloc --gres=gpu:1 --cpus-per-task=1 --mem=16G --time=00:30:00

   # Exit on error
   set -e

   module --quiet purge
   module load anaconda/3
   module load cuda/11.7

   ENV_NAME=pytorch

   ## Create the environment (see the example):
   conda create --yes --name $ENV_NAME python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.7 --channel pytorch --channel nvidia
   # Install as many packages as possible with Conda:
   conda install --yes --name $ENV_NAME tqdm --channel conda-forge
   # Activate the environment:
   conda activate $ENV_NAME
   # Install the rest of the packages with pip:
   pip install rich
   conda env export --no-builds --from-history --file environment.yaml

Exit the interactive job once the environment has been created. Then, the
example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh
