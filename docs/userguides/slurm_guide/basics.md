---
title: Understanding Slurm
description: Ask for a resource allocation and launch tasks on the cluster.
---

# Understanding Slurm

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Get Started with the Cluster__](../../../getting_started/)
    { .card }

    ---
    Obtain a Mila account, enable cluster access and MFA, install `uv` and
    `milatools`, configure SSH access and connect to the cluster for the first
    time.

-   [:material-run-fast:{ .lg .middle } __What is a computer cluster?__](../../../technical_reference/general_theory/cluster_parts/)
    { .card }

    ---
    Understand some notions of what a cluster is. This is an introduction to login nodes, compute nodes and storage nodes.

&nbsp;

</div>


## What this guide covers
* Discovering the Slurm jobs, steps and tasks
* Launching multiple tasks through an interactive job
* Launch multiple tasks from a script

## Key concepts

### Jobs, steps and tasks
Recurrent entities when we speak of Slurm are jobs, steps and tasks. If we want to keep a simple scheme, we could say that:

* a job can have multiple steps
* a step can run multiple tasks.

??? info "More information"
    A **job** can have multiple **steps**. These steps are created when srun is called from inside the job and the slurmstepd created will run the command passed to srun. Each step will have one instance of slurmstepd created per node in the step and each instance of slurmstepd can run multiple tasks in the same step.

    [Source - Slurm documentation](https://slurm.schedmd.com/job_launch.html)

### Login nodes and compute nodes
We will not dive into details here, because these concepts have been explained in [What is a computer cluster?](../../../technical_reference/general_theory/cluster_parts/), but to sum up some notions, we focus on the two following types of nodes:

| Type of node | Use |
| ------------ | --- |
| Login node   | They are used to connect to the cluster and manage your jobs |
| Compute node | This is where the jobs run, the allocation requested when a job is launched is provided from them |


### Commands

The three Slurm commands we focus on are:

| Command  | Entity created | Description | From where call the command? |
| -------- | -------------- | ----------- | ---------------------------- |
| `sbatch` | Non-interactive job | Submit a batch script to Slurm | From a login node |
| `salloc` | Interactive job | Obtain a Slurm job allocation (a set of nodes), execute a command, and then release the allocation when the command is finished | From a login node |
| `srun`   | Step :material-information-outline:{ title="srun can also be used to directly submit jobs, but we don't recommend it" } | Run tasks | From a job |


Submitting tasks is done through two steps:

1. Request a resource allocation by submitting a job (`sbatch` or `salloc`)
2. Launch commands by launching tasks from this resource allocation (`srun`)



## Discover Slurm through an interactive job

**1. Connect to your favorite cluster (see [this section](../../../getting_started/#verify-your-connection) for more information on the connection)**

=== "Steps"
    
    Open a terminal and launch the command:
    ```bash
    ssh mila
    ```

=== "More details"

    Here, we arbitrary choose to connect to the Mila cluster. The command `ssh mila` works thanks to the configuration we previously set inside `~/.ssh/config`. This could be done by `mila init` (see [the Getting Started guide](../../getting_started/)).


**2. Submit a job**

=== "Steps"

    Submitting a job is like booking an allocation: you request which resources you want (GPU, CPU, node, memory), setting your experiment conditions. The Slurm scheduler is then in charge to provide you an allocation.

    ```bash
    salloc --ntasks=4 --nodes=2 --mem=2G --time=00:30:00
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    salloc: --------------------------------------------------------------------------------------------------
    salloc: # Using default long-cpu partition (CPU-only)
    salloc: --------------------------------------------------------------------------------------------------
    salloc: Pending job allocation 9311988
    salloc: job 9311988 queued and waiting for resources

    salloc: Granted job allocation 9311988
    salloc: Nodes cn-f[001-002] are ready for job
    ```
    </div>

    Once the allocation is done, you get some information about your job:
    
    * you know what your Job ID is (9311988 in this example)
    * you know on which nodes your allocation is (cn-f001 and cn-f002 in this example).


    Congrats! You now have a resource allocation.

=== "More details"

    * `salloc` means this is an interactive job
    * `--ntasks` means that `srun` will invoke 4 tasks
    * `--nodes` means 2 nodes are requested for the previously mentioned tasks to run on
    * `--mem` aspecify the real memory required per node. We could also set `--mem-per-gpu` or `--mem-per-cpu`
    * `--time` asks for an allocation of 30min. It is a good practice to set it because an interactive job can last until one week, and it is a common mistake to forget to leave an interactive job.

    See [salloc documentation](https://slurm.schedmd.com/salloc.html) for more information.



**3. Understand where we are**

=== "Steps"

    By running the command `hostname`, you can see "where" the process calling the command runs:

    ```bash
    hostname
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    cn-f001.server.mila.quebec
    ```
    </div>

    Now, let's try to run steps and tasks by using `srun`:

    ```bash
    srun hostname
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    cn-f002.server.mila.quebec
    cn-f002.server.mila.quebec
    cn-f002.server.mila.quebec
    cn-f001.server.mila.quebec
    ```
    </div>

    Each task returned its own result for the `hostname` command.

    In this example, we can see that:
    
    * three tasks have been launched on the node `cn-f002`
    * one task has been launched on the node `cn-f001`

=== "More details"

    * Note on the command:
        * We used `srun hostname`, presenting the format `srun <command>`. `srun` can also take parameters in the format `srun <parameters> <command>`. See [srun documentation](https://slurm.schedmd.com/srun.html) for more details.

    * Notes on the result:
        * The `hostname` command has been called four times because we ask for four tasks while submitting the job through `salloc`.
        * By running our four tasks with `srun`, we can see that they are not necessarily evenly spread among the nodes.


## Launch a non-interactive job

In this section, we reproduce the same example as before (same parameters and same command (`hostname`)) and submit the job through the `sbatch` command.

**1. Connect to your favorite cluster**

```bash
ssh mila
```

**2. Write the script**

=== "Steps"
    You could either:

    * write the script directly on the login node (in `$SCRATCH` repository and its children repositories)
        ```bash
        cd $SCRATCH
        vim job.sh
        ```
    * or write it on your local computer and copy it to the scratch directory through:
        ```bash
        scp job.sh mila:/network/scratch/s/user.name
        ```

        by using, instead of `user.name` your own name.


    The content of `job.sh` is:

    ```bash
    #!/bin/bash
    #SBATCH --ntasks=4
    #SBATCH --nodes=2
    #SBATCH --mem=2G

    hostname
    ```

=== "More details"
    We add in the beginning of the script the same parameters we used while running `salloc` for our interactive job. You can notice that we did not add the time here. The parameter exists, and if it is not set, the partition's default time limit is used.


**3. Launch the command**

From the login node, run:

```
sbatch job.sh
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
sbatch: --------------------------------------------------------------------------------------------------
sbatch: # Using default long-cpu partition (CPU-only)
sbatch: --------------------------------------------------------------------------------------------------
Submitted batch job 9321166
```
</div>


Now that the job is submitted, all that is left to do is waiting for it to be scheduled. You can see it status by running the [`squeue`](https://slurm.schedmd.com/squeue.html) command:

```
squeue --me
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
JOBID     USER    PARTITION           NAME  ST START_TIME             TIME NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) COMMENT
9321166 user.name long-cpu,lon      job.sh  PD N/A                    0:00     2    4        N/A      2G  (Priority) (null)
```
</div>

Here, the allocation is requested by sbatch based on the script parameters. Once it is ready, the script is automatically executed (ie the job is running), the allocation is freed at the end of the job.


**4. Retrieve the results**

Once the job is finished, its output can be retrieved by reading the file `slurm-<JOB_ID>.out`. (In our case, the file name is: `slurm-9321166.out`). This can be changed by using the parameter [`--output`](https://slurm.schedmd.com/sbatch.html#OPT_output).



The content of the output in our example is:
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
cn-f001.server.mila.quebec
```
</div>

We explain in [this guide](tasks_communication.md) why it does not display the hostname of the four tasks, as it happened in our interactive job, but only one instead.

---

## Key concepts

Job
:   Global commands executed in a requested resources allocation.

Task
:   Set of commands running on an allocation part. A job can contain multiple tasks.

---

## Next step

Now that we can launch multiple tasks with their own variables (such as hostname), let's learn how to make them communicate.

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Synchronizing multiple tasks__](tasks_communication.md)
    { .card }

    ---
    Synchronize the output of multiple tasks on different node.

&nbsp;

</div>
