The workload manager
********************

Once connected to a login node, presumably with SSH, you can issue a
job execution request to what is called the workload manager. The
workload manager used at Mila and Compute Canada clusters is called
:ref:`Slurm <slurmpage>`.  The workload manager's main role is to find a
place to run your program in what is simply called a *job*. This
"place" is in fact one of many computers synchronised to the manager
which are called :ref:`The compute nodes`.

In fact it's a bit trickier than that, but we'll stay at this abstraction level
for now.

.. include:: Theory_cluster_slurm.rst
