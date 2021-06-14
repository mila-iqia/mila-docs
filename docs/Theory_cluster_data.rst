Processing data
===============

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



