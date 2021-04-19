SLURM Architecture
===================

On the cloud clusters, only the login nodes and the SLURM controller are static.
The compute nodes are deployed on-demand (``STATE=CLOUD`` in the SLURM config) i.e. each time you submit a job,
SLURM allocate a node to your job and send a request to a middleware to start the node.
It can take up to *5 minutes to boot* a node depending of the load, that is why batch (``sbatch``) jobs are
encouraged. If you need to start interactive jobs, you should look at wrapping a interactive shell inside a
batch job (jupyterhub/notebook, smux/tmux, etc...). If your allocated node did not boot
in the expected time, your job will be requeued and you will not loose your priority in the queue.


Compute Nodes
-------------
Compute Nodes on every provider are running on preemptible VMs i.e. your node can be
disconnected at any time (GCP reboot preemptible nodes every 24h but send a signal that can be caught...todo).


Automatic routing
^^^^^^^^^^^^^^^^^^

For convenience, an automatic routing script has been implemented to allocate the right-sized node to your job.

.. note::
    For now only single-node computation are allowed in the beta phase

.. tip::
    No need to specify a queue as your job is automatically routed to the queue containing nodes with the required number of GPUs/RAM
    SLURM jobs run with 1 task, 1 node, and the number of CPUs matching the number of CPUs-per-GPU on the specified cloud provider



+--------------------------------+----------------------------------+
| Submission                     |  Equivalent partition            |
+================================+==================================+
| ``sbatch --gres=gpu:v100:2``   |  Partiton=azureV100x2Cloud       |
+--------------------------------+----------------------------------+
| ``sbatch -c6 --mem=64G``       |  Partiton=azureGeneralCPUCloud   |
+--------------------------------+----------------------------------+
| ``sbatch --gres=gpu:k80:1``    |  Partiton=azureK80x1Cloud        |
+--------------------------------+----------------------------------+

Partitions
----------

Current partitions on Azure/GCP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To display the available queues

.. prompt:: bash $

   sinfo -l --format="%P %z %m %G %N"



====================    =====================   ======  ==========  =============================
Azure                   Sockets/Cores/Threads   RAM     Gres        Nodes
====================    =====================   ======  ==========  =============================
azureComputeCPUCloud    2:4:2                   32000   -           azure-c8-32g-kobe[1-18]
azureGeneralCPUCloud    2:4:2                   64000   -           azure-c8-64g-yokohama[1-8]
azureMemoryCPUCloud     2:4:2                   128000  -           azure-c8-128g-tokyo[1-8]
azureV100x1Cloud        1:6:1                   112000  gpu:v100:1  azure-gv100-canberra[1-12]
azureV100x2Cloud        1:12:1                  224000  gpu:v100:2  azure-gv100x2-brisbane[1-4]
azureV100x4Cloud        2:12:1                  448000  gpu:v100:4  azure-gv100x4-melbourne[1-2]
azureK80x1Cloud         1:6:1                   56000   gpu:k80:1   azure-gk80-manchester[1-48]
azureK80x2Cloud         1:12:1                  112000  gpu:k80:2   azure-gk80x2-birmingham[1-8]
azureK80x4Cloud         2:12:1                  224000  gpu:k80:4   azure-gk80x4-london[1-8]
====================    =====================   ======  ==========  =============================

.. note::
    The limit in the number of nodes per family is currently higher than what is reflected in the partitions
    but more nodes will be added gradually

Quotas
------
In SLURM, every job is recorded and consumes resources. To limit the consumption per-user according to
what you have been granted in terms of $, each account is associated with
a ``TRESMins`` maximum that you can access using ``sshare``.

.. note::
   Each element of a node (CPU, Mem, GPU) contributes to the TRESMins and there is a limit per element e.g.
   a job of 60mins running on 6 cpus and 1 V100 GPU will consume *6x60xCPUmins* and *1x60xGRESmins*

Checking your Quota/Limits

.. prompt:: bash $

      sshare -A $LOGNAME -o Account,GrpTRESMIns,GrpTRESRaw%200

For convenience, a custom script is available in your ``PATH`` called ``squotas``

.. prompt:: bash $

      squotas
