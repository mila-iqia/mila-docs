# SLURM commands guide

## Some definitions
### Jobs, job steps and tasks

  * A **job** is allocated defined resources (GPUs, CPU cores, RAM) for a defined time.
    It is created with either `sbatch` (non-interactive) or `salloc` (interactive).
    A job can be subdivided into steps.
  * A **job step** sub-allocates from the resources of the job. Job steps are created
    with the `srun` wrapper command when it is called from inside a job[^1].
    By default a job step will consume all resources allocated to the job,
    but this can be changed.

    !!! tip
        Steps most naturally map to phases of a job[^2]: For example,

          * Job step `.1` might correspond to a stage-in of the dataset to `$SLURM_TMPDIR`
          * Job step `.2` might run your distributed parallel Python program, and
          * Job step `.3` might stage out the results to `$SCRATCH`.

  * A job step can be further subdivided into **tasks**. A task is a specific instance
    of the command created in multiple copies by `srun`.
    Every task has an associated ID within its step, a number `$SLURM_PROCID` also called
    a "rank" between `0` and `$SLURM_NPROCS`-1 inclusive.

    !!! tip
        A task most naturally maps to one of the main processes in a distributed program
        (but see [^3]). While a job/job step can be multi-node, each task will run on one
        and only one node.


In machine-learning work, it is very important to _think carefully_ about _how many
times should something be done_. Some things must be done once **per job**; others,
once **per node**, once **per GPU**, or even once **per CPU core**.
The number of tasks in a job step is a very important tool in deciding this.
One way this can play out, when reusing the three-job-step example above:

<details>

<summary>Job step .1 – Dataset Stage-In</summary>

<i>
Staging in a dataset should be done once per node, because </i>
<code>$SLURM_TMPDIR</code>
<i> is a filesystem private to each node,
but can be accessed by all processes of that node.
</i>

```bash
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES tar -xf $SCRATCH/my_dataset.tar -C $SLURM_TMPDIR
```

</details>

<details open>

<summary>Job step .2 – Distributed Processing</summary>

<i>
The distributed program must run one main process per GPU, because with DDP,
each process is mean to manage only one GPU.
</i>

```bash
srun                      python   my_script.py # Best; inherits job's defaults
srun --ntasks-per-gpu=1   python   my_script.py # Good, maybe redundant
srun --ntasks-per-node=4  python   my_script.py # Worse; Hardcodes GPU count.

# If using torchrun/accelerate and autodetect GPU count:
srun --ntasks-per-node=1  torchrun my_script.py
```

These are not the only valid choices
<sup id="fnref:3"><a class="footnote-ref" href="#fn:3">3</a></sup>
<sup id="fnref:4"><a class="footnote-ref" href="#fn:4">4</a></sup>.

</details>

<details>

<summary>Job step .3 – Results Stage-Out</summary>

<code>$SCRATCH</code><i> is shared by every node and every process can see it.
To avoid collisions, some jobs might arrange for only only one task overall to
write out the job's results.
</i>

```bash
srun --ntasks-per-node=1 --ntasks=1 cp -a $SLURM_TMPDIR/results $SCRATCH/results/$SLURM_JOBID
```

</details>

[^1]: Outside of a job, srun creates a job, not a job step; This is an overloaded use of
    `srun` that is confusing and that we discourage.

[^2]: Job steps usually run sequentially (subdivision in time), because a plain
    `srun` will use all the resources of a job, but job steps can also run in parallel
    if `srun` is given arguments that request few resources _(subdivision in space and
    resources)_ that two steps can proceed in parallel.

[^3]: By far the most common configurations for the tasks of a job's main step will be:

      - 1 task per GPU (`--ntasks-per-gpu=1`), when launching every process of a
        distributed program directly with `srun`, or
      - 1 task per node (`--ntasks-per-node=1`), when launching `torchrun`/`accelerate`
        once per node, and allowing these processes to decide the appropriate number of
        children processes themselves.

    They lead to different strategies for orchestrating the distributed program's
    processes, and different environment-variable contents.

    A less common configuration is `--ntasks-per-gpu=N` to do _N:1_ packing of
    processes onto a single GPU that they all share.

[^4]: Even within a task, there are further levels of subdivision and parallelism possible:
    A process can create subprocesses ("workers"), and these processes can create multiple
    threads. However, these concepts are out of scope for SLURM and are more closely
    related to the _Linux kernel's_ concept of threads, processes.

    The _SLURM_ concept of "task" must also not be confused with the _Linux kernel's_
    usage of the same terminology to refer to the basic unit of scheduling - roughly a thread.

## Basic Usage

The SLURM [documentation](https://slurm.schedmd.com/documentation.html)
provides extensive information on the available commands to query the cluster
status or submit jobs.

Below are some basic examples of how to use SLURM.

### Submitting jobs

#### Batch job

In order to submit a batch job, you have to create a script containing the main
command(s) you would like to execute on the allocated resources/nodes.

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=100Gb

srun uv run python my_script.py
```

Your job script is then submitted to SLURM with [`sbatch`](https://slurm.schedmd.com/sbatch.html).

```console
$ sbatch job_script
sbatch: Submitted batch job 4323674
```

The *working directory* of the job will be the one where your executed `sbatch`.

!!! tip
    Slurm directives can be specified on the command line alongside ``sbatch`` or
    inside the job script with a line starting with ``#SBATCH``.


#### Interactive job

Workload managers usually run batch jobs to avoid having to watch its
progression and let the scheduler run it as soon as resources are available. If
you want to get access to a shell while leveraging cluster resources, you can
submit an interactive jobs where the main executable is a shell with the
[srun](https://slurm.schedmd.com/srun.html) or [salloc](https://slurm.schedmd.com/salloc.html) commands.

```console
salloc
```

Will start an interactive job on the first node available with the default
resources set in SLURM (1 task/1 CPU).  `srun` accepts the same arguments as
`sbatch` with the exception that the environment is not passed.

!!! tip
    To pass your current environment to an interactive job, add
    ``--preserve-env`` to ``srun``.

`salloc` can also be used and is mostly a wrapper around `srun` if provided
without more info but it gives more flexibility if for example you want to get
an allocation on multiple nodes.

#### Job submission arguments

In order to accurately select the resources for your job, several arguments are
available. The most important ones are:

| Argument                        | Description                                                                |
| ------------------------------- | -------------------------------------------------------------------------- |
| `-n`, `--ntasks=<number>`       | The number of task in your script, usually =1                              |
| `-c`, `--cpus-per-task=<ncpus>` | The number of cores for each task                                          |
| `-t`, `--time=<time>`           | Time requested for your job                                                |
| `--mem=<size[units]>`           | Memory requested for all your tasks                                        |
| `--gres=<list>`                 | Select generic resources such as GPUs for your job: `--gres=gpu:GPU_MODEL` |

!!! tip
    Always consider requesting the adequate amount of resources to improve the
    scheduling of your job (small jobs always run first).

### Checking job status

To display *jobs* currently in queue, use `squeue` and to get only your jobs type

<!-- todo: `squeue --me` also does the same thing and is quicker to write.  -->
```bash
$ squeue -u $USER
 JOBID   USER          NAME    ST  START_TIME         TIME NODES CPUS TRES_PER_NMIN_MEM NODELIST (REASON) COMMENT
 133     my_username   myjob   R   2019-03-28T18:33   0:50     1    2        N/A  7000M node1 (None) (null)
```

!!! note
    The maximum number of jobs able to be submitted to the system per user is 1000 (MaxSubmitJobs=1000)
    at any given time from the given association. If this limit is reached, new submission requests
    will be denied until existing jobs in this association complete.

### Removing a job

To cancel your job simply use `scancel`

```bash
scancel 4323674
```

### Partitioning


See the [list of Mila cluster partitions](../../technical_reference/clusters/mila/nodes). To request an unkillable job with 1 GPU, 4 CPUs, 10G of RAM and
12h of computation do:

```console
sbatch --gres=gpu:1 -c 4 --mem=10G -t 12:00:00 --partition=unkillable <job.sh>
```

You can also make it an interactive job using `salloc`:

```console
salloc --gres=gpu:1 -c 4 --mem=10G -t 12:00:00 --partition=unkillable
```

The Mila cluster has many different types of nodes/GPUs. To request a specific
type of node/GPU, you can add specific feature requirements to your job
submission command.

To access those special nodes you need to request them explicitly by adding the
flag `--constraint=<name>`.  The full list of nodes in the Mila Cluster can be
accessed at [Node profile description](../../technical_reference/clusters/mila/nodes).

*Examples:*

To request a machine with 2 GPUs using NVLink, you can use

```console
sbatch -c 4 --gres=gpu:2 --constraint=nvlink
```

To request a DGX system with 8 A100 GPUs, you can use

```console
sbatch -c 16 --gres=gpu:8 --constraint="dgx&ampere"
```

#### Information on partitions/nodes

[`sinfo`](https://slurm.schedmd.com/sinfo.html) provides most of the
information about available nodes and partitions/queues to submit jobs to.

Partitions are a group of nodes usually sharing similar features. On a
partition, some job limits can be applied which will override those asked for a
job (i.e. max time, max CPUs, etc...)

To display available *partitions*, simply use


```console
$ sinfo
PARTITION AVAIL TIMELIMIT NODES STATE  NODELIST
batch     up     infinite     2 alloc  node[1,3,5-9]
batch     up     infinite     6 idle   node[10-15]
cpu       up     infinite     6 idle   cpu_node[1-15]
gpu       up     infinite     6 idle   gpu_node[1-15]
```

To display available *nodes* and their status, you can use


```console
$ sinfo -N -l
NODELIST    NODES PARTITION STATE  CPUS MEMORY TMP_DISK WEIGHT FEATURES REASON
node[1,3,5-9]   2 batch     allocated 2    246    16000     0  (null)   (null)
node[2,4]       2 batch     drain     2    246    16000     0  (null)   (null)
node[10-15]     6 batch     idle      2    246    16000     0  (null)   (null)
...
```

And to get statistics on a job running or terminated, use `sacct` with some of
the fields you want to display


```console
$ sacct --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,nnodes,ncpus,nodelist,workdir -u $USER
     User        JobID    JobName  Partition      State  Timelimit               Start                 End    Elapsed   NNodes      NCPUS        NodeList              WorkDir
--------- ------------ ---------- ---------- ---------- ---------- ------------------- ------------------- ---------- -------- ---------- --------------- --------------------
my_usern+ 2398         run_extra+      batch    RUNNING 130-05:00+ 2019-03-27T18:33:43             Unknown 1-01:07:54        1         16 node9           /home/mila/my_usern+
my_usern+ 2399         run_extra+      batch    RUNNING 130-05:00+ 2019-03-26T08:51:38             Unknown 2-10:49:59        1         16 node9           /home/mila/my_usern+
```

Or to get the list of all your previous jobs, use the `--start=YYYY-MM-DD` flag. You can check `sacct(1)` for further information about additional time formats.

```console
sacct -u $USER --start=2019-01-01
```

[`scontrol`](https://slurm.schedmd.com/scontrol.html) can be used to
provide specific information on a job (currently running or recently terminated)

```console
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
```

Or more info on a node and its resources

```console
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
```

## Useful Commands

```console title="Get an interactive job and give you a shell. (ssh like) CPU only"
salloc
```

```console title="Get an interactive job with one GPU, 2 CPUs and 12000 MB RAM"
salloc --gres=gpu:1 -c 2 --mem=12000
```

```console title="start a batch job (same options as salloc)"
sbatch
```
```console title="Re-attach a dropped interactive job"
sattach \--pty <jobid>.0
```
```console title="status of all nodes"
sinfo
```
```console title="List GPU type and FEATURES that you can request"
sinfo -Ogres:27,nodelist,features -tidle,mix,alloc
```
```console title="(Custom) List available gpus"
savail
```
```console title="Cancel a job"
scancel <jobid>
```
```console title="summary status of all active jobs"
squeue
```
```console title="summary status of all YOUR active jobs"
squeue -u $USER
```
```console title="summary status of a specific job"
squeue -j <jobid>
```
```console title="status of all jobs including requested resources (see the SLURM squeue doc for all output options)"
squeue -Ojobid,name,username,partition,state,timeused,nodelist,gres,tres
```
```console title="Detailed status of a running job"
scontrol show job <jobid>
```
```console title="Get the node where a finished job ran"
sacct -j <job_id> -o NodeList
```
```console title="Find info about old jobs"
sacct -u $USER -S <start_time> -E <stop_time>
```
```console title="List of current and recent jobs"
sacct -oJobID,JobName,User,Partition,Node,State
```

### Special GPU requirements

Specific GPU *architecture* and *memory* can be easily requested through the
`--gres` flag by using either

* `--gres=gpu:architecture:number`
* `--gres=gpu:memory:number`
* `--gres=gpu:model:number`

*Example*:

To request 1 GPU with *at least* 48GB of memory use

```console
sbatch -c 4 --gres=gpu:48gb:1
```

The full list of GPU and their features can be accessed [here](../../technical_reference/clusters/mila/nodes).

### Example script

Here is a `sbatch` script that follows good practices on the Mila cluster:

```bash
#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/<u>/<username>/slurm-%j.out  # Write the log on scratch

# 1. Copy your dataset on the compute node
cp /network/datasets/<dataset> $SLURM_TMPDIR

# 2. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
srun uv run python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

# 3. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/<to_save> /network/scratch/<u>/<username>/
```
