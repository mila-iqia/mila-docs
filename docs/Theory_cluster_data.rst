Processing data
===============

For processing large amounts of data common for deep learning, either
for dataset preprocessing or training, several techniques exist. Each
has typical uses and limitations.


Data parallelism
----------------

The first technique is called **data parallelism** (aka task
parallelism in formal computer science). You simply run lots of
processes each handling a portion of the data you want to
process. This is by far the easiest technique to use and should be
priviledged whenever possible. A common example of this is
hyperparameter optimisation.

For really small computations the time to setup multiple processes
might be longer than the processing time and lead to waste. This can
be addressed by bunching up some of the processes toghether by doing
sequential processing of sub-partitions of the data.

For the cluster systems it is also inadvisable to launch thousands of
jobs and even if each job would run for a reasonable amount of time
(several minutes at minimum), it would be best to make larger groups
until the amount of jobs is in the low hundreds at most.

Finally another thing to keep in mind is that the transfer bandwidth
is limited between the filesystems and the compute nodes and if you
run too many jobs using too much data at once they may end up not
being any faster because they will spend their time waiting for data
to arrive.


Model parallelism
-----------------

The second technique is called **model parallelism** (which doesn't
have a single equivalent in formal computer science). It is used
mostly when a single instance of a model will not fit in a computing
resource (such as the GPU memory being too small for all the
parameters).

In this case the model is split across layer (or some other logical
arrangement) and the tasks must communicate intermediate results with
each other to arrive at a final result.

This is generally harder but necessary to work with larger, more
powerful models like GPT.

Communication concerns
----------------------

The main difference of these two approaches is the need for
communication between the multiple processes. Some common training
methods, like stochastic gradient descent sit somewhere between the
two, because they require some communication, but not a lot. Most
people classify it as data parallelism since it sits closer to that
end.

In general for data parallelism tasks or tasks that communicate
infrequently it doesn't make a lot of difference where the processes
sit because the communcation bandwitdth and latency will not have a
lot of impact on the time it takes to complete the job.  The
individual tasks can generally be scheduled indepandantly.

On the contrary for model parallelism you need to pay more attention
to where your tasks are.  In this case it is usually required to use
the facilities of the job scheduler to group the tasks so that they
are on the same machine or machines that are closely linked to ensure
optimal communication.  What is the best allocation depends on the
specific cluster architecture available and the technologies it
support (such as `InfiniBand <https://en.wikipedia.org/wiki/InfiniBand>`_,
`RDMA <https://en.wikipedia.org/wiki/Remote_direct_memory_access>`_,
`NVLink <https://en.wikipedia.org/wiki/NVLink>`_ or others)


Filesystems
===========

Clusters have different types of file systems to support different data
storage use cases. We differentiate them by name. You'll hear or read about
file systems such as "home", "scratch" or "project" and so on.

Most of these file systems are are provided in a way which is globally
available to all nodes in the cluster. Software or data required by jobs can
be accessed from any node on the cluster.
(See :ref:`Mila <milacluster_storage>` or :ref:`CC <cc_storage>` for more
information on available file systems)

Different file systems have different performance levels. For instance, backed
up file-systems ( such as ``$PROJECT`` ) provide more space and can handle
large files but cannot sustain highly parallel accesses typically required
for high speed model training.

Each compute node has local file systems ( of which ``$SLURM_TMPDIR`` ) that
are usually more efficient but any data remaining on these will be erased at
the end of the job execution for the next job to come along.

