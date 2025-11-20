.. NOTE: This file is auto-generated from examples/frameworks/jax_setup/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _jax_setup:

Jax Setup
=========

**Prerequisites**: (Make sure to read the following before using this example!)

* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`uv`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax_setup>`_


**job.sh**


.. code:: bash

   #!/bin/bash
   #SBATCH --ntasks=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=1
   #SBATCH --gpus-per-task=l40s:1
   #SBATCH --mem-per-gpu=16G
   #SBATCH --time=00:15:00

   # Exit on error
   set -e

   # Echo time and hostname into log
   echo "Date:     $(date)"
   echo "Hostname: $(hostname)"

   # Execute Python script
   # Use `uv run --offline` on clusters without internet access on compute nodes.
   uv run python main.py


**pyproject.toml**

.. code:: toml

   [project]
   name = "jax-setup"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.rst"
   requires-python = ">=3.11,<3.14"
   dependencies = ["jax[cuda12]>=0.5.3"]


**main.py**

.. code:: python

   import jax
   import jax.extend.backend


   def main():
       device_count = len(jax.local_devices(backend="gpu"))
       print(f"Jax default backend:         {jax.extend.backend.get_backend().platform}")
       print(f"Jax-detected #GPUs:          {device_count}")

       if device_count == 0:
           print("    No GPU detected, not printing device names.")
       else:
           for i, device in enumerate(jax.local_devices(backend="gpu")):
               print(f"    GPU {i}:      {device.device_kind}")


   if __name__ == "__main__":
       main()


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
