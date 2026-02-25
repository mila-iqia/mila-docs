The workload manager
********************

On a cluster, users don't have direct access to the compute nodes but
instead connect to a login node and add jobs to the workload manager
queue. Whenever there are resources available to execute these jobs
they will be allocated to a compute node and run, which can be
immediately or after a wait of up to several days.


Anatomy of a job
================

A job is comprised of a number of steps that will run one after the
other. This is done so that you can schedule a sequence of processes
that can use the results of the previous steps without having to
manually interact with the scheduler.

Each step can have any number of tasks which are groups of processes
that can be scheduled independently on the cluster but can run in
parallel if there are resources available. The distinction between
steps and tasks is that multiple tasks, if they are part of the same
step, cannot depend on results of other tasks because there are no
guarantees on the order in which they will be executed.

Finally each process group is the basic unit that is scheduled in the
cluster. It comprises of a set of processes (or threads) that can run
on a number of resources (CPU, GPU, RAM, ...) and are scheduled
together as a unit on one or more machines.

Each of these concepts lends itself to a particular use. For multi-gpu
training in AI workloads you would use one task per GPU for data
paralellism or one process group if you are doing model
parallelism. Hyperparameter optimisation can be done using a
combination of tasks and steps but is probably better left to a
framework outside of the scope of the workload manager.

If this all seems complicated, you should know that all these things
do not need to always be used. It is perfectly acceptable to sumbit
jobs with a single step, a single task and a single process.


Understanding the queue
=======================

The available resources on the cluster are not infinite and it is the
workload manager's job to allocate them. Whenever a job request comes
in and there are not enough resources available to start it
immediately, it will go in the queue.

Once a job is in the queue, it will stay there until another job
finishes and then the workload manager will try to use the newly freed
resources with jobs from the queue. The exact order in which the jobs
will start is not fixed, because it depends on the local policies
which can take into account the user priority, the time since the job
was requested, the amount of resources requested and possibly other
things. There should be a tool that comes with the manager where you
can see the status of your queued jobs and why they remain in the
queue.


About partitions
================

The workload manager will divide the cluster into partitions according
to the configuration set by the admins. A partition is a set of
machines typically reserved for a particular purpose. An example might
be CPU-only machines for preprocessing setup as a separate partition.
It is possible for multiple partitions to share resources.

There will always be at least one partition that is the default
partition in which jobs without a specific request will go. Other
partitions can be requested, but might be restricted to a group of
users, depending on policy.

Partitions are useful for a policy standpoint to ensure efficient use
of the cluster resources and avoid using up too much of one resource
type blocking use of another. They are also useful for heterogenous
clusters where different hardware is mixed in and not all software is
compatible with all of it (for example x86 and POWER cpus).


Exceding limits (preemption and grace periods)
==============================================

To ensure a fair share of the computing resources for all, the workload
manager establishes limits on the amount of resources that a single
user can use at once. These can be hard limits which prevent running
jobs when you go over or soft limits which will let you run jobs, but
only until some other job needs the resources.

Admin policy will determine what those exact limits are for a
particular cluster or user and whether they are hard or soft limits.

The way soft limits are enforced is using preemption, which means that
when another job with higher priority needs the resources that your
job is using, your job will receive a signal that it needs to save its
state and exit. It will be given a certain amount of time to do this
(the grace period, which may be 0s) and then forcefully terminated if
it is still running.

Depending on the workload manager in use and the cluster configuration
a job that is preempted like this may be automatically rescheduled to
have a chance to finish or it may be up to the job to reschedule
itself.

The other limit you can encounter with a job that goes over its
declared limits. When you schedule a job, you declare how much
resources it will need (RAM, CPUs, GPUs, ...). Some of those may have
default values and not be explicitely defined. For certain types of
devices, like GPUs, access to units over your job limit is made
unavailable. For others, like RAM, usage is monitored and your job
will be terminated if it goes too much over. This makes it important
to ensure you estimate resource usage accurately.


.. This should be somewhere else, but I don't know where.

Mila information
================

**Mila** as well as `Digital Research Alliance of Canada
<https://docs.alliancecan.ca/wiki/Technical_documentation>`_ use the workload
manager `Slurm <https://slurm.schedmd.com/documentation.html>`_ to schedule and
allocate resources on their infrastructure.

**Slurm** client commands are available on the login nodes for you to submit
jobs to the main controller and add your job to the queue. Jobs are of 2 types:
*batch* jobs and *interactive* jobs.

For practical examples of Slurm commands on the Mila cluster, see :ref:`Running
your code`.
