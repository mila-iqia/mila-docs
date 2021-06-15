Running your code
=================

Since we don't have many GPUs on the cluster, resources must be shared as fairly as possible.
The ``--partition=/-p`` flag of SLURM allows you to set the priority you need for a job.
Each job assigned with a priority can preempt jobs with a lower priority:
``unkillable > main > long``. Once preempted, your job is killed without notice and is automatically re-queued
on the same partition until resources are available. (To leverage a different preemption mechanism,
see the :ref:`Handling preemption <advanced_preemption>`)

========================== ========================== ============ ============
Flag                         Max Resource Usage       Max Time     Note
========================== ========================== ============ ============
--partition=unkillable       1 GPU, 6 CPUs, mem=32G     2 days
--partition=main             2 GPUs, 8 CPUs, mem=48G    2 days
--partition=long             no limit of resources      7 days
========================== ========================== ============ ============

For instance, to request an unkillable job with 1 GPU, 4 CPUs, 10G of RAM and 12h of computation do:

.. prompt:: bash $

    sbatch --gres=gpu:1 -c 4 --mem=10G -t 12:00:00 --partition=unkillable <job.sh>

You can also make it an interactive job using ``salloc``:

.. prompt:: bash $

    salloc --gres=gpu:1 -c 4 --mem=10G -t 12:00:00 --partition=unkillable


The Mila cluster has many different types of nodes/GPUs. To request a specific type of node/GPU, you can
add specific feature requirements to your job submission command.

To access those special nodes you need to request them explicitly by adding the flag ``--constraint=<name>``.
The full list of nodes in the Mila Cluster can be accessed :ref:`Node profile
description`.

*Example:*

To request a Power9 machine

.. prompt:: bash $

    sbatch -c 4 --constraint=power9


To request a machine with 2 GPUs using NVLink, you can use

.. prompt:: bash $

    sbatch -c 4 --gres=gpu:2 --constraint=nvlink


=========================================== =================================================================================
Feature                                        Particularities
=========================================== =================================================================================
x86_64 (Default)                               Regular nodes
Power9                                         :ref:`Power9 <power9_nodes>` CPUs (incompatible with x86_64 software)
12GB/16GB/24GB/32GB/48GB                       Request a specific amount of *GPU* memory
maxwell/pascal/volta/tesla/turing/kepler       Request a specific *GPU* architecture
nvlink                                         Machine with GPUs using the NVLink technology
=========================================== =================================================================================


.. note::

	You don't need to specify *x86_64* when you add a constraint as it is added by default ( ``nvlink`` -> ``x86_64&nvlink`` )



Special GPU requirements
------------------------

Specific GPU *architecture* and *memory* can be easily requested through the ``--gres`` flag by using either

* ``--gres=gpu:architecture:memory:number``
* ``--gres=gpu:architecture:number``
* ``--gres=gpu:memory:number``
* ``--gres=gpu:model:number``


*Example:*

To request a Tesla GPU with *at least* 16GB of memory use

.. prompt:: bash $

    sbatch -c 4 --gres=gpu:tesla:16gb:1

The full list of GPU and their features can be accessed :ref:`here <node_list>`.


CPU-only jobs
-------------

Since the priority is given to the usage of GPUs, CPU-only jobs have a low priority and can only consume **4 cpus maximum per node**.
The partition for CPU-only jobs is named ``cpu_jobs`` and you can request it with ``-p cpu_jobs`` or if you don't specify any GPU, you will be
automatically rerouted to this partition.


Script Example
--------------

Here is a ``sbatch`` script that follows good practices on the Mila cluster:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --partition=unkillable                      # Ask for unkillable job
    #SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
    #SBATCH --gres=gpu:1                          # Ask for 1 GPU
    #SBATCH --mem=10G                             # Ask for 10 GB of RAM
    #SBATCH --time=3:00:00                        # The job will run for 3 hours
    #SBATCH -o /network/tmp1/<user>/slurm-%j.out  # Write the log on tmp1

    # 1. Load the required modules
    module --quiet load anaconda/3

    # 2. Load your environment
    conda activate <env_name>

    # 3. Copy your dataset on the compute node
    cp /network/data/<dataset> $SLURM_TMPDIR

    # 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
    #    and look for the dataset into $SLURM_TMPDIR
    python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

    # 5. Copy whatever you want to save on $SCRATCH
    cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/

