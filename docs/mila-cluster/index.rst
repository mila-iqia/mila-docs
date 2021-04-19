.. _milacluster:




Mila Cluster
############

.. highlight:: bash

The Mila Cluster is an heterogeneous cluster. It uses :ref:`SLURM<slurmpage>` to schedule jobs.

.. contents:: Mila Cluster
   :depth: 2
   :local:


.. toctree::
   :maxdepth: 2

   monitoring/index
   FAQs <faq/index>

.. toctree::
   :maxdepth: 2
   :includehidden:

   nodes



Account Creation
""""""""""""""""

To access the Mila Cluster clusters, you will need an account. Please contact Mila systems administrators if you don't have it already. Our IT support service is available here: https://it-support.mila.quebec/


Login
"""""

You can access the Mila cluster via ssh:

.. prompt:: bash $

    ssh <user>@login.server.mila.quebec -p 2222

4 login nodes are available and accessible behind a Load-Balancer.
At each connection, you will be redirected to the least loaded login-node.
Each login node can be directly accessed via: ``login-X.login.server.mila.quebec`` on port ``2222``.

The login nodes support the following authentication mechanisms: ``publickey,keyboard-interactive``.
If you would like to set an entry in your ``.ssh/config`` file, please use the following recommendation:


.. code-block:: bash

        Host HOSTALIAS
            User YOUR-USERNAME
            Hostname login.server.mila.quebec
            PreferredAuthentications publickey,keyboard-interactive
            Port 2222
            ServerAliveInterval 120
            ServerAliveCountMax 5


The RSA, DSA and ECDSA fingerprints for Mila's login nodes are:

.. code-block:: bash

    SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8 (ECDSA)
    SHA256:XvukABPjV75guEgJX1rNxlDlaEg+IqQzUnPiGJ4VRMM (DSA)
    SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o (RSA)
    SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg (ED25519)


Launching Jobs
""""""""""""""

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
The full list of nodes in the Mila Cluster can be accessed :ref:`here <node_list>`.

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~

Since the priority is given to the usage of GPUs, CPU-only jobs have a low priority and can only consume **4 cpus maximum per node**.
The partition for CPU-only jobs is named ``cpu_jobs`` and you can request it with ``-p cpu_jobs`` or if you don't specify any GPU, you will be
automatically rerouted to this partition.


Modules
"""""""

Many software, such as Python and Conda, are already compiled and available on the cluster through the ``module`` command and
its sub-commands. In particular, if you with to use ``Python 3.7`` you can simply do:

.. prompt:: bash $

    module load python/3.7

Please look at the :ref:`Module <modules>` and :ref:`Python <python>` sections for more details.

.. _milacluster_storage:

Storage
"""""""

..
   TODO : Cette table doit être mise à jour.


=============================== ========================================= ======================= ================
Path                             Usage                                     Quota (Space/Files)     Auto-cleanup*
=============================== ========================================= ======================= ================
/home/mila/<u>/<username>/         * Personal user space                       200G/1000K
                                   * Specific libraries/code
/miniscratch/                      * Temporary job results                                          3 months
/network/projects/<groupname>/     * Long-term project storage                 200G/1000K
/network/data1/                    * Raw datasets (read only)
/network/datasets/                 * Curated raw datasets (read only)
$SLURM_TMPDIR                      * High speed disk for                                           at job end
                                     temporary job results
=============================== ========================================= ======================= ================

* The ``home`` folder is appropriate for codes and libraries which are small and read once, as well as the experimental results you wish to keep (e.g. the weights of a network you have used in a paper).
* The ``data1`` folder should only contain **compressed** datasets.
* To request the addition of a dataset into ``datasets``, `this form <https://docs.google.com/forms/d/e/1FAIpQLSf3XyKpvMaIHV2MaDBbFIVJCCcWLpo2HTjkQ94f-ebMRpY97Q/viewform?usp=sf_link/>`_ should be filled ``datasets``.
* ``$SLURM_TMPDIR`` points to the local disk of the node on which a job is running. It should be used to copy the data on the node at the beginning of the job and write intermediate checkpoints. This folder is cleared after each job.

* **Auto-cleanup** is applied on files not touched during the specified period


.. warning:: We currently do not have a backup system in the lab. You should backup whatever you want to keep on your laptop, google drive, etc.



Script Example
""""""""""""""

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
