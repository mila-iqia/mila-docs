---
title: Synchronizing multiple tasks
description: A quick example of multiple tasks synchronizing their output.
---

# Synchronizing multiple tasks

## Before you begin

<div class="grid cards" markdown>

-   [:material-lightbulb-alert-outline:{ .lg .middle } __Understanding Slurm__](basics.md)
    { .card }

    ---
    Use an interactive job to run multiple tasks.

-   [:material-monitor-eye:{ .lg .middle } __Monitor and manage jobs__](monitor_manage.md)
    { .card }

    ---
    Track jobs through the queue, inspect and cancel them, and read their
    output.

</div>

## What this guide covers

* Launching multiple tasks with `sbatch`
* Sharing variables between tasks

## Concept of this example

In plain terms, this example runs four tasks across two nodes. Each task holds
one number equal to its rank (0, 1, 2 and 3). The tasks add their numbers
together, and only the first task prints the total (6). Reaching a single total
requires the tasks to communicate, which is what this example demonstrates.

This example launches a job (using `job_***.sh`) that runs one or more tasks
(whose instructions are stored in `main_jax.py` or `main_torch.py`) using
libraries (defined in `pyproject.toml`).

Each example is based on three files:

| File | Description |
| ---- | ----------- |
| `job_***.sh` | Bash script used to request an allocation and launch a job (which itself runs multiple tasks based on the requested `--ntasks`) |
| `main_***.py` | Python script containing the instructions the tasks execute. This example uses either Jax (with the script `main_jax.py`) or Pytorch (with the script `main_torch.py`) |
| `pyproject.toml` | Configuration file used to handle the libraries `uv` fetches. A separate `pyproject.toml` could be used for each example (Jax and Torch), but both libraries are gathered in one to simplify this guide |


### Introducing the different files

=== "job_torch.sh"
    ```bash
    #!/bin/bash
    #SBATCH --ntasks=4
    #SBATCH --nodes=2
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    #SBATCH --time=00:01:00

    # These environment variables are read by the distributed runtime and
    # should ideally be set before running the python script, or at the very
    # beginning of the python script.

    # Master address is the hostname of the first node in the job.
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" \
         | head -n 1)
    # Derive a per-job port from the last 4 digits of the job ID
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
    export WORLD_SIZE=$SLURM_NTASKS

    srun uv run python main_torch.py
    ```

=== "job_jax.sh"
    ```bash
    #!/bin/bash
    #SBATCH --ntasks=4
    #SBATCH --nodes=2
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    #SBATCH --time=00:01:00

    # These environment variables are read by the distributed runtime and
    # should ideally be set before running the python script, or at the very
    # beginning of the python script.

    # Master address is the hostname of the first node in the job.
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" \
         | head -n 1)
    # Derive a per-job port from the last 4 digits of the job ID
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
    export WORLD_SIZE=$SLURM_NTASKS

    srun uv run python main_jax.py
    ```

??? info "In-depth script explanation on `job_***.sh`"
    **Headers for the resources allocation**

    The `#SBATCH` header lines request the resource allocation: 4 tasks across 2
    nodes, 1 CPU per task, 8G of memory and a 1-minute time limit.

    **Environment variables**

    The environment variables `MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE` are
    defined here and can be retrieved in each task. `MASTER_PORT` derives a
    per-job port from the last 4 digits of the job ID (a value in the 10000 to
    19999 range), so that jobs running at the same time do not collide on the
    same port. In Python, retrieving an environment variable value is done as
    follows:
    ```python
    import os # Retrieving an environment variable is done through os.environ

    MASTER_ADDR = os.environ["MASTER_ADDR"]
    ```


    **Running the tasks**

    `srun uv run python main_***.py`

    * The command `srun` creates tasks. The number of tasks is determined by the
      `--ntasks` parameter of the allocation. Here, 4 tasks were requested, so
      the command runs 4 tasks in parallel. These tasks run the command
      following `srun`, so each task runs `uv run python main_torch.py` or `uv
      run python main_jax.py`.
    * `uv run` is used to ease the environment set up for the tasks. For more
      information, read the [`uv` guide on portability](../python_uv.md). It is
      followed by the name of the script to run in this environment.


=== "main_torch.py"
    ```python
    import os

    import torch
    import torch.distributed

    RANK = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    NODE_INDEX = int(os.environ["SLURM_NODEID"])

    # Defined in the sbatch script, hostname of the first node in the job.
    # export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_ADDR = os.environ.get("MASTER_ADDR")
    # Get a unique port for this job based on the job ID
    MASTER_PORT = os.environ.get("MASTER_PORT")


    def main():

        group = torch.distributed.init_process_group(
            init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
            world_size=WORLD_SIZE,
            rank=RANK,
            backend="gloo" # https://docs.pytorch.org/docs/main/distributed.html#which-backend-to-use
        )

        x = torch.tensor([float(RANK)], dtype=torch.float32)
        print(f"\n[Node {NODE_INDEX} | Rank {RANK}] x={x[0]}")

        total = torch.clone(x)
        torch.distributed.reduce(
            total, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if NODE_INDEX == 0 and RANK == 0: # The complete sum lands on the first task of the first node
            print(f"sum={total[0]}")
        torch.distributed.destroy_process_group(group)


    if __name__ == "__main__":
        main()
    ```

=== "main_jax.py"
    ```python
    import jax
    import jax.distributed
    import os

    RANK = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    NODE_INDEX = int(os.environ["SLURM_NODEID"])

    def main():
        jax.config.update("jax_platforms", "cpu")
        jax.distributed.initialize() # Prepare JAX for execution on multi-host GPU, must be called before performing any JAX computations

        x = jax.numpy.array([float(RANK)], dtype=jax.numpy.float32) # For each task, x depends on RANK, which is different between all tasks
        print(f"\n[Node {NODE_INDEX} | Rank {RANK}] x={x[0]}")

        # Sum x across all tasks.
        total = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
        if NODE_INDEX == 0 and RANK == 0:
            print(f"sum={total[0]}")


    if __name__ == "__main__":
        main()
    ```

??? info "In-depth script explanation on `main_***.py`"
    **Pytorch and Jax**

    This guide is based on two open source examples

    * [Pytorch](https://pytorch.org/) is a deep-learning library.
    * [Jax](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) is a
      library for array-oriented numerical computation.

    **Environment variables**

    Each file retrieves the Slurm environment variables `SLURM_PROCID`,
    `SLURM_NTASKS` and `SLURM_NODEID`. Unlike the environment variables defined
    previously (`MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE`), these
    environment variables are specific to each task. More common Slurm
    environment variables are listed in [the technical
    reference](../../technical_reference/general_theory/slurm.md).

    ```python
    RANK = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    NODE_INDEX = int(os.environ["SLURM_NODEID"])
    ```

    === "What happens in Torch script"
        1. Initialize: in Torch, a group is defined
        2. Create a value, different for each task

            The created value is based on the RANK, which is specific to each
            task

        3. Compute their sum

    === "What happens in Jax script"
        1. Initialize: this function is specific to Jax
        2. Create a value, different for each task

            The created value is based on the RANK, which is specific to each
            task

        3. Compute their sum [see Jax Lax parallel
           operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators)

    The final sum is printed from the first task of the first node (NODE_INDEX=0
    and RANK=0). This is the task where all the `x` values have been collected.
    On the other tasks, `total` holds a partial result.


=== "pyproject.toml (for Pytorch)"
    ```toml
    [project]
    name = "multitasks-demo"
    version = "0.1.0"
    description = "Using Jax and Torch to illustrate a multitask example"
    requires-python = ">=3.11,<3.14"
    dependencies = ["torch>=2.7.1"]
    ```

=== "pyproject.toml (for Jax)"
    ```toml
    [project]
    name = "multitasks-demo"
    version = "0.1.0"
    description = "Using Jax and Torch to illustrate a multitask example"
    requires-python = ">=3.11,<3.14"
    dependencies = ["jax>=0.5.3"]
    ```


??? info "In-depth explanation on `pyproject.toml`"
    `pyproject.toml` is a configuration file used by packaging tools (`uv` in
    this case) ([More info on `pyproject.toml`
    files](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)).
    The value of dependencies contains information about the libraries used in
    this example. `torch` is used with the `main_torch.py` script, and `jax`
    with the `main_jax.py` script. To use only one of them, delete the unused
    library from the `pyproject.toml` file.

### Launching the example

1. Create the three files on the cluster

    === "VSCode"

        Open the project on a compute node with `mila code`, or pick `mila-cpu`
        in the Remote-SSH dropdown, then create `job_***.sh`, `main_***.py` and
        `pyproject.toml` in the VSCode explorer. See
        [VSCode](../../toolbox/VSCode.md) and the [Get Started
        guide](../../getting_started/index.md).

    === "Terminal"

        Connect to the cluster with `ssh mila`, then create the files in
        `$SCRATCH` with an editor such as `vim`.

        ```bash
        ssh mila
        ```

2. Launch the job

    In the VSCode integrated terminal (or a login-node terminal), submit the
    job:

    === "Launch Torch example"

        ```bash
        sbatch job_torch.sh
        ```

    === "Launch Jax example"

        ```bash
        sbatch job_jax.sh
        ```


3. (Optional) Check the job status

    ```bash
    squeue --me
    ```

    See [Monitor and manage jobs](monitor_manage.md) for how to read the
    output, inspect the job once it finishes, and cancel it if needed.

4. Retrieve the results

    When the resources have been allocated and the script has run, an output
    file has been created: it is by default called `slurm-{JOB_ID}.out`, with
    `JOB_ID` being the ID of the job which has run.

    === "Torch script results"

        <div class="result" style="border:None; padding:0" markdown>
        ``` linenums="0"
        [Node 0 | Rank 0] x=0.0
        sum=6.0
        [Node 0 | Rank 1] x=1.0
        [Node 1 | Rank 3] x=3.0
        [Node 1 | Rank 2] x=2.0
        ```
        </div>

    === "Jax script results"

        <div class="result" style="border:None; padding:0" markdown>
        ``` linenums="0"
        [Node 1 | Rank 2] x=2.0
        [Node 0 | Rank 1] x=1.0
        [Node 1 | Rank 3] x=3.0
        [Node 0 | Rank 0] x=0.0
        sum=6.0
        ```
        </div>

    For each example, the ranks of the tasks (that is, their `x` values) are
    respectively 0, 1, 2 and 3. Their sum, retrieved on [Node 0 | Task 0], is
    therefore 6.

## Next step

<div class="grid cards" markdown>

-   [:material-multicast:{ .lg .middle } __Launch many jobs from the same shell script__](../../examples/good_practices/launch_many_jobs/index.md)
    { .card }

    ---
    Good practice to run the same experiment with different arguments.

&nbsp;

</div>
