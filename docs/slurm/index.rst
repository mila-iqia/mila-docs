.. highlight:: bash

.. _slurmpage:

Quick Start
===========

Resource sharing on a supercomputer/cluster is orchestrated by a resource manage/job scheduler.
Users submit jobs, which are scheduled and allocated resources (CPU time, memory, GPUs, etc.) by the resource manager, if
the resources are available the job can start otherwise it will be placed in queue.

On a cluster, users don't have direct access to the compute nodes but instead connect to a login node to pass the commands they
would like to execute in a script for the workload manager to execute.

**Mila** as well as **Compute Canada** use the workload manager `Slurm <https://slurm.schedmd.com/documentation.html>`_
to schedule and allocate resources on their infrastructure.

**Slurm** client commands are available on the login nodes for you to submit jobs to the main controller and add your job
to the queue. Jobs are of 2 types: *batch* jobs and *interactive* jobs.



Basic Usage
===========

The SLURM `documentation <https://slurm.schedmd.com/documentation.html>`_ provides extensive information on the
available commands to query the cluster status or submit jobs.

Below are some basic examples of how to use SLURM.



Submitting jobs
---------------

Batch job
^^^^^^^^^
In order to submit a batch job, you have to create a script containing the main command(s) you would like to execute on
the allocated resources/nodes.

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --output=job_output.txt
    #SBATCH --error=job_error.txt
    #SBATCH --ntasks=1
    #SBATCH --time=10:00
    #SBATCH --mem=100Gb

    module load python/3.5
    python my_script.py


Your job script is then submitted to SLURM with ``sbatch`` (`ref. <https://slurm.schedmd.com/sbatch.html>`__)

.. prompt:: bash $, auto

    $ sbatch job_script
    sbatch: Submitted batch job 4323674

The *working directory* of the job will be the one where your executed ``sbatch``.

.. tip::
   Slurm directives can be specified on the command line alongside ``sbatch`` or inside the job script with a line
   starting with ``#SBATCH``.


Interactive job
^^^^^^^^^^^^^^^

Workload managers usually run batch jobs to avoid having to watch its progression and let the scheduler
run it as soon as resources are available. If you want to get access to a shell while leveraging cluster resources,
you can submit an interactive jobs where the main executable is a shell with the
``srun/salloc`` (`srun <https://slurm.schedmd.com/srun.html>`_/`salloc <https://slurm.schedmd.com/salloc.html>`_) commands

.. prompt:: bash $

    salloc

will start an interactive job on the first node available with the default resources set in SLURM (1 task/1 CPU).
``srun`` accepts the same arguments as ``sbatch`` with the exception that the environment is not passed.

.. tip::
   To pass your current environment to an interactive job, add ``--preserve-env`` to ``srun``.

``salloc`` can also be used and is mostly a wrapper around ``srun`` if provided without more info but it gives more flexibility
if for example you want to get an allocation on multiple nodes.



Job submission arguments
^^^^^^^^^^^^^^^^^^^^^^^^

In order to accurately select the resources for your job, several arguments are available. The most important ones are:

=============================     ====================================================================================
  Argument                         Description
=============================     ====================================================================================
-n, --ntasks=<number>              The number of task in your script, usually =1
-c, --cpus-per-task=<ncpus>        The number of cores for each task
-t, --time=<time>                  Time requested for your job
--mem=<size[units]>                Memory requested for all your tasks
--gres=<list>                      Select generic resources such as GPUs for your job: ``--gres=gpu:GPU_MODEL``
=============================     ====================================================================================

.. tip::
   Always consider requesting the adequate amount of resources to improve the scheduling of your job (small jobs always run first).


Managing jobs
--------------

Checking job status
^^^^^^^^^^^^^^^^^^^

To display *jobs* currently in queue, use ``squeue`` and to get only your jobs type

.. prompt:: bash $, auto

    $ squeue -u $USER
    JOBID   USER          NAME    ST  START_TIME         TIME NODES CPUS TRES_PER_NMIN_MEM NODELIST (REASON) COMMENT
    133     my_username   myjob   R   2019-03-28T18:33   0:50     1    2        N/A  7000M c1-8g-tiny1 (None) (null)



Removing a job
^^^^^^^^^^^^^^^

To cancel your job simply use ``scancel``

.. prompt:: bash $

    scancel 4323674


Cluster status
--------------

Information on partitions/nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sinfo`` (`ref. <https://slurm.schedmd.com/sinfo.html>`__) provides most of the information
about available nodes and partitions/queues to submit jobs to.

Partitions are a group of nodes usually sharing similar features. On a partition, some
job limits can be applied which will override those asked for a job (i.e. max time, max CPUs, etc...)

To display available *partitions*, simply use

.. prompt:: bash $, auto

    $ sinfo
    PARTITION AVAIL TIMELIMIT NODES STATE  NODELIST
    batch     up     infinite     2 alloc  node[1,3,5-9]
    batch     up     infinite     6 idle   node[10-15]
    cpu       up     infinite     6 idle   cpu_node[1-15]
    gpu       up     infinite     6 idle   gpu_node[1-15]


To display available *nodes* and their status, you can use

.. prompt:: bash $, auto

    $ sinfo -N -l
    NODELIST    NODES PARTITION STATE  CPUS MEMORY TMP_DISK WEIGHT FEATURES REASON
    node[1,3,5-9]   2 batch     allocated 2    246    16000     0  (null)   (null)
    node[2,4]       2 batch     drain     2    246    16000     0  (null)   (null)
    node[10-15]     6 batch     idle      2    246    16000     0  (null)   (null)
    ...

and to get statistics on a job running or terminated, use ``sacct`` with some of the fields you want to display

.. prompt:: bash $, auto

    $ sacct --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,nnodes,ncpus,nodelist,workdir -u username
     User        JobID    JobName  Partition      State  Timelimit               Start                 End    Elapsed   NNodes      NCPUS        NodeList              WorkDir
    --------- ------------ ---------- ---------- ---------- ---------- ------------------- ------------------- ---------- -------- ---------- --------------- --------------------
    username 2398         run_extra+ azureComp+    RUNNING 130-05:00+ 2019-03-27T18:33:43             Unknown 1-01:07:54        1         16 node9         /home/mila/username+
    username 2399         run_extra+ azureComp+    RUNNING 130-05:00+ 2019-03-26T08:51:38             Unknown 2-10:49:59        1         16 node9         /home/mila/username+


or to get the list of all your previous jobs, use the ``--start=####`` flag

.. prompt:: bash

    sacct -u my_username --start=20190101


``scontrol`` (`ref. <https://slurm.schedmd.com/scontrol.html>`__) can be used to provide
specific information on a job (currently running or recently terminated)

.. prompt:: bash $, auto

    $ scontrol show job 43123
    JobId=43123 JobName=python_script.py
    UserId=my_username(1500000111) GroupId=student(1500000000) MCS_label=N/A
    Priority=645895 Nice=0 Account=my_username QOS=normal
    JobState=RUNNING Reason=None Dependency=(null)
    Requeue=1 Restarts=3 BatchFlag=1 Reboot=0 ExitCode=0:0
    RunTime=2-10:41:57 TimeLimit=130-05:00:00 TimeMin=N/A
    SubmitTime=2019-03-26T08:47:17 EligibleTime=2019-03-26T08:49:18
    AccrueTime=2019-03-26T08:49:18
    StartTime=2019-03-26T08:51:38 EndTime=2019-08-03T13:51:38 Deadline=N/A
    PreemptTime=None SuspendTime=None SecsPreSuspend=0
    LastSchedEval=2019-03-26T08:49:18
    Partition=slurm_partition AllocNode:Sid=login-node-1:14586
    ReqNodeList=(null) ExcNodeList=(null)
    NodeList=node2
    BatchHost=node2
    NumNodes=1 NumCPUs=16 NumTasks=1 CPUs/Task=16 ReqB:S:C:T=0:0:*:*
    TRES=cpu=16,mem=32000M,node=1,billing=3
    Socks/Node=* NtasksPerN:B:S:C=1:0:*:* CoreSpec=*
    MinCPUsNode=16 MinMemoryNode=32000M MinTmpDiskNode=0
    Features=(null) DelayBoot=00:00:00
    OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
    WorkDir=/home/mila/my_username
    StdErr=/home/mila/my_username/slurm-43123.out
    StdIn=/dev/null
    StdOut=/home/mila/my_username/slurm-43123.out
    Power=

or more info on a node and its resources

.. prompt:: bash $, auto

    $ scontrol show node node9
    NodeName=node9 Arch=x86_64 CoresPerSocket=4
    CPUAlloc=16 CPUTot=16 CPULoad=1.38
    AvailableFeatures=(null)
    ActiveFeatures=(null)
    Gres=(null)
    NodeAddr=10.252.232.4 NodeHostName=mila20684000000 Port=0 Version=18.08
    OS=Linux 4.15.0-1036 #38-Ubuntu SMP Fri Dec 7 02:47:47 UTC 2018
    RealMemory=32000 AllocMem=32000 FreeMem=23262 Sockets=2 Boards=1
    State=ALLOCATED+CLOUD ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
    Partitions=slurm_partition
    BootTime=2019-03-26T08:50:01 SlurmdStartTime=2019-03-26T08:51:15
    CfgTRES=cpu=16,mem=32000M,billing=3
    AllocTRES=cpu=16,mem=32000M
    CapWatts=n/a
    CurrentWatts=0 LowestJoules=0 ConsumedJoules=0
    ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s



Useful Commands
----------------




+----------------------------------------------------------+-----------------------------------------------------------------------------+
| Command                                                  | Description                                                                 |
+==========================================================+=============================================================================+
| salloc                                                   | Get an interactive job and give you a shell. (ssh like) CPU only            |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| salloc --gres=gpu -c 2 --mem=12000                       | Get an interactive job with one GPU, 2 CPUs and 12000 MB RAM                |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sbatch                                                   | start a batch job (same options as salloc)                                  |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sattach --pty <jobid>.0                                  | Re-attach a dropped interactive job                                         |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sinfo                                                    | status of all nodes                                                         |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sinfo -Ogres:27,nodelist,features -tidle,mix,alloc       | List GPU type and FEATURES that you can request                             |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| savail                                                   | (Custom) List available gpu                                                 |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| scancel <jobid>                                          | Cancel a job                                                                |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| squeue                                                   | summary status of all active jobs                                           |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| squeue -u $USER                                          | summary status of all YOUR active jobs                                      |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| squeue -j <jobid>                                        | summary status of a specific job                                            |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| squeue -Ojobid,name,username,partition,                  |  status of all jobs including requested                                     |
| state,timeused,nodelist,gres,tres                        |  resources (see the SLURM squeue doc for all output options)                |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| scontrol show job <jobid>                                | Detailed status of a running job                                            |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sacct -j <job_id> -o NodeList                            | Get the node where a finished job ran                                       |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sacct -u $USER -S <start_time> -E <stop_time>            | Find info about old jobs                                                    |
+----------------------------------------------------------+-----------------------------------------------------------------------------+
| sacct -oJobID,JobName,User,Partition,Node,State          | List of current and recent jobs                                             |
+----------------------------------------------------------+-----------------------------------------------------------------------------+




Fairshare/Priority
-------------------

TODO
