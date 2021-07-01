.. highlight:: bash

.. _slurmpage:

SLURM
-----

Resource sharing on a supercomputer/cluster is orchestrated by a resource
manage/job scheduler.  Users submit jobs, which are scheduled and allocated
resources (CPU time, memory, GPUs, etc.) by the resource manager, if the
resources are available the job can start otherwise it will be placed in queue.

On a cluster, users don't have direct access to the compute nodes but instead
connect to a login node to pass the commands they would like to execute in a
script for the workload manager to execute.

**Mila** as well as **Compute Canada** use the workload manager `Slurm
<https://slurm.schedmd.com/documentation.html>`_ to schedule and allocate
resources on their infrastructure.

**Slurm** client commands are available on the login nodes for you to submit
jobs to the main controller and add your job to the queue. Jobs are of 2 types:
*batch* jobs and *interactive* jobs.

For practical examples of SLURM commands on the Mila cluster, see :ref:`Running
your code`.
