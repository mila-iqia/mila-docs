Processing data
***************

For processing large amounts of data common for deep learning, either
for dataset preprocessing or training, several techniques exist. Each
has typical uses and limitations.

Data parallelism
================

The first technique is called **data parallelism** (aka task
parallelism in formal computer science). You simply run lots of
processes each handling a portion of the data you want to
process. This is by far the easiest technique to use and should be
favored whenever possible. A common example of this is
hyperparameter optimisation.

For really small computations the time to setup multiple processes
might be longer than the processing time and lead to waste. This can
be addressed by bunching up some of the processes together by doing
sequential processing of sub-partitions of the data.

For the cluster systems it is also inadvisable to launch thousands of
jobs and even if each job would run for a reasonable amount of time
(several minutes at minimum), it would be best to make larger groups
until the amount of jobs is in the low hundreds at most.

Finally another thing to keep in mind is that the transfer bandwidth
is limited between the filesystems (see :ref:`Filesystem concerns`)
and the compute nodes and if you run too many jobs using too much data
at once they may end up not being any faster because they will spend
their time waiting for data to arrive.


Model parallelism
=================

The second technique is called **model parallelism** (which doesn't
have a single equivalent in formal computer science). It is used
mostly when a single instance of a model will not fit in a computing
resource (such as the GPU memory being too small for all the
parameters).

In this case, the model is split into its constituent parts, each
processed independently and their intermediate results communicated
with each other to arrive at a final result.

This is generally harder but necessary to work with larger, more
powerful models like GPT.

Communication concerns
======================

The main difference of these two approaches is the need for
communication between the multiple processes. Some common training
methods, like stochastic gradient descent sit somewhere between the
two, because they require some communication, but not a lot. Most
people classify it as data parallelism since it sits closer to that
end.

In general for data parallelism tasks or tasks that communicate
infrequently it doesn't make a lot of difference where the processes
sit because the communication bandwidth and latency will not have a
lot of impact on the time it takes to complete the job.  The
individual tasks can generally be scheduled independently.

On the contrary for model parallelism you need to pay more attention
to where your tasks are.  In this case it is usually required to use
the facilities of the workload manager to group the tasks so that they
are on the same machine or machines that are closely linked to ensure
optimal communication.  What is the best allocation depends on the
specific cluster architecture available and the technologies it
support (such as `InfiniBand <https://en.wikipedia.org/wiki/InfiniBand>`_,
`RDMA <https://en.wikipedia.org/wiki/Remote_direct_memory_access>`_,
`NVLink <https://en.wikipedia.org/wiki/NVLink>`_ or others)


Filesystem concerns
===================

When working on a cluster, you will generally encounter several
different filesystems.  Usually there will be names such as 'home',
'scratch', 'datasets', 'projects', 'tmp'.

The reason for having different filesystems available instead of a
single giant one is to provide for different use cases. For example,
the 'datasets' filesystem would be optimized for fast reads but have
slow write performance. This is because datasets are usually written
once and then read very often for training.

Different filesystems have different performance levels. For instance, backed
up filesystems (such as ``$PROJECT`` in Digital Research Alliance of Canada
clusters) provide more space and can handle large files but cannot sustain
highly parallel accesses typically required for high speed model training.

The set of filesystems provided by the cluster you are using should be
detailed in the documentation for that cluster and the names can
differ from those above. You should pay attention to their recommended
use case in the documentation and use the appropriate filesystem for
the appropriate job. There are cases where a job ran hundreds of times
slower because it tried to use a filesystem that wasn't a good fit for
the job.

One last thing to pay attention to is the data retention policy for
the filesystems. This has two subpoints: how long is the data kept
for, and are there backups.

Some filesystems will have a limit on how long they keep their
files. Typically the limit is some number of days (like 90 days) but
can also be 'as long as the job runs' for some.

As for backups, some filesystems will not have a limit for data, but
will also not have backups. For those it is important to maintain a
copy of any crucial data somewhere else. The data will not be
purposefully deleted, but the filesystem may fail and lose all or part
of its data. If you have any data that is crucial for a paper or your
thesis keep an additional copy of it somewhere else.
