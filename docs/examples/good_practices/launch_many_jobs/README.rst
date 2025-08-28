.. NOTE: This file is auto-generated from examples/good_practices/launch_many_jobs/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _launch_many_jobs:

Launch many jobs from same shell script
=======================================

Sometimes you may want to run the same job with different arguments.
For example, you may want to launch an experiment using a few different learning rates.
This example shows an easy way to do this.


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_
* `examples/distributed/single_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu>`_


**job.sh**

Compared to the :ref:`single_gpu_job` example, here we use the ``$@`` bash directive
to pass command-line arguments down to the Python script.

This makes it very easy to submit multiple jobs, each with different values!

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/launch_many_jobs>`_

.. code:: diff

    # distributed/single_gpu/job.sh -> good_practices/launch_many_jobs/job.sh
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=00:15:00

    set -e  # exit on error.
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"

    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
    cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

   -# Execute Python script
   +# Execute Python script, passing down the command-line arguments.
   +# This allows you to reuse the same submission script when submitting multiple jobs,
   +# each with different arguments.
    # Use `uv run --offline` on clusters without internet access on compute nodes.
   -uv run python main.py
   +uv run python main.py "$@"


**Running this example**

You can run this example just like the :ref:`single_gpu_job` example, but you can now
also pass command-line arguments directly when submitting the job with ``sbatch``!

For example:

.. code-block:: bash

    $ sbatch job.sh --learning-rate 0.1
    $ sbatch job.sh --learning-rate 0.5
    $ sbatch job.sh --weight-decay 1e-3


Next steps
^^^^^^^^^^

These next examples build on top of this one and show how to properly launch lots of jobs for hyper-parameter sweeps:
* :ref:`Using SLURM Job Arrays to launch lots of jobs <slurm_job_arrays>`
* :ref:`Running more effective Hyper-Parameter Sweeps with Orion <hpo_with_orion>`
