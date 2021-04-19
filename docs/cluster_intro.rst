What is a computer Cluster ?
----------------------------

   A computer cluster is a set of loosely or tightly connected computers that
   work together so that, in many respects, they can be viewed as a single
   system.

   `Wikipedia <https://en.wikipedia.org/wiki/Computer_cluster>`__

In order to provide high performance computation capabilities, clusters can
combine hundreds to thousands of computers, called *nodes*, which are all
inter-connected with a high-performance communication network. Most nodes are
designed for high-performance computations, but clusters can also use
specialized nodes to offer parallel file systems, databases, login nodes and
even the cluster scheduling functionality as pictured in the image below.

.. image:: cluster_overview2.png

All clusters typically run on GNU/Linux distributions. Hence a minimum
knowledge of GNU/Linux and BASH is usually required to use them. See the
following `tutorial <https://docs.computecanada.ca/wiki/Linux_introduction>`_
for a rough guide on getting started with Linux.


The Login Nodes
^^^^^^^^^^^^^^^

To execute computing processes on a cluster, you must first connect to a login
node. These login nodes are the entry point to most clusters. Another entry
point to some clusters such as the Mila cluster is the JupyterHub WEB
interface, but we'll read about that later.

For now let's return to the subject of this section; Login nodes. To connect to
these, you would typically use a remote shell connection. The most usual tool
to do so is SSH. You'll hear and read a lot about this tool. Imagine it as a
very long (and somewhat magical) extension cord which connects the computer you
are using now, such as your laptop, to a remote computer's terminal shell. If
you followed through with the tutorial in the section above, you already know
what a terminal shell is.


The job scheduler
^^^^^^^^^^^^^^^^^

Once connected to a login node, presumably with SSH, you can issue a job
execution request to what is called the job scheduler. The job scheduler used
at Mila and Compute Canada clusters is called SLURM (:ref:`slurm <slurmpage>`).
The job scheduler's main role is to find a place to run your program in what is
simply called : *a job*. This "place" is in fact one of many computers
synchronised to the scheduler which are called : *Compute Nodes*.

In fact it's a bit trickier than that, but we'll stay at this abstraction level
for now.

Compute and Login nodes : Different nodes for different uses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is important to note here the difference in intended uses between the
compute nodes and the login nodes.

While the Compute Nodes are meant for heavy computation, the Login Nodes are
not. In the field of artificial intelligence, you will usually be on the hunt
for GPUs. The compute nodes are the ones with GPU capacity.

The login nodes however are used by everyone who uses the cluster and care must
be taken not to overburden these nodes. Consequently, only very short and light
processes should be run on these otherwise the cluster may become inaccessible.
In other words, please refrain from executing long or compute intensive
processes on login nodes because it affects all other users. In some cases, you
will also find that doing so might get you into trouble.


File systems
^^^^^^^^^^^^

Clusters have different types of file systems to support different data
storage use cases. We differentiate them by name. You'll hear or read about
file systems such as "home", "scratch" or "project" and so on.

Most of these file systems are are provided in a way which is globally available to all nodes in the cluster. Software or data required by jobs can be accessed from any node on the cluster. (See :ref:`Mila <milacluster_storage>` or :ref:`CC <cc_storage>` for more information on available file systems)

Different file systems have different performance levels. For instance, backed
up file-systems ( such as ``$PROJECT`` ) provide more space and can handle large
files but cannot sustain highly parallel accesses typically required for high speed model training.

Each compute node has local file systems ( of which ``$SLURM_TMPDIR`` ) that
are usually more efficient but any data remaining on these will be erased at
the end of the job execution for the next job to come along.


Resources Available at Mila
----------------------------

This table contains the computational resources Mila has access to.

.. Warning:: This table is outdated and needs to be revisited.

================================ ================================ ====
Cluster                          CPUs                             GPUs
================================ ================================ ====
:ref:`Mila <milacluster>`                                         248
-------------------------------- -------------------------------- ----
:ref:`Beluga <beluga>`           34k                              688 V100
-------------------------------- -------------------------------- ----
:ref:`Cedar <cedar>`             27k                              584 P100
-------------------------------- -------------------------------- ----
:ref:`Graham <graham>`           36k                              320 P100
-------------------------------- -------------------------------- ----
Helios                                                            216 k80
-------------------------------- -------------------------------- ----
Niagara                          60k
-------------------------------- -------------------------------- ----
Total                            134k                             2040
================================ ================================ ====

:ref:`Mila Cluster <milacluster>`
   This cluster is to be used for regular development and relatively small
   number of jobs (< 5)

:ref:`Compute Canada Clusters <cc_clusters>` The clusters Beluga, Cedar,
   Graham, Helios and Niagara are clusters provided by Compute Canada on which we
   have allocations. These are to be used for many jobs, multi-nodes and/or
   multi-GPU jobs as well as long running jobs
