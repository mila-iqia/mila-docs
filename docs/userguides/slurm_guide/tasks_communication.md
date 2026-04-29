---
title: Synchronizing multiple tasks
description: A quick example of multiple tasks synchronizing their output.
---

# Synchronizing multiple tasks

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Understanding Slurm__](basics.md)
    { .card }

    ---
    Use an interactive job to run multiple tasks.

&nbsp;

</div>

## What this guide covers
* Launching multiple tasks with `sbatch`
* Sharing tasks variables of different tasks

## Concept of this example

In this guide, we launch a job (using `job.sh`) which will run one or more tasks (whose instructions are stored in `main_jax.py` or `main_torch.py`) using libraries (defined in `pyproject.toml`).

Thus, each example is based on three files:

| File | Description |
| ---- | ----------- |
| `job.sh` | Bash script used to request an allocation and launch a job (which itself runs multiple tasks based on the requested `--ntasks`) |
| `main_***.py` | Python script containing the instructions the tasks execute. In this example, we either use Jax (with the script `main_jax.py`) or Pytorch (with the script `main_torch.py`) |
| `pyproject.toml` | Configuration file used to handle the libraries `uv` is gonna get. We could have done one `pyproject.toml` for each example (Jax and Torch), but we gathered the two libraries in one to simplify this guide |


### Introducing the different files


=== "job.sh"
    ```bash
    #!/bin/bash
    #SBATCH --ntasks=4
    #SBATCH --nodes=2
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G

    # These environment variables are used by torch.distributed and should ideally be set
    # before running the python script, or at the very beginning of the python script.
    
    # Master address is the hostname of the first node in the job.
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    # Get a unique port for this job based on the job ID
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
    export WORLD_SIZE=$SLURM_NTASKS

    srun uv run "$@"
    ```

??? info "In-depth script explaination on `job.sh`"
    **Headers for the resources allocation**

    These are the header and the parameters we request for the resources allocation.
    ```bash
    #!/bin/bash
    #SBATCH --ntasks=4
    #SBATCH --nodes=2
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    ```

    **Environment variables**

    The environment variables `MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE` are defined here and can be retrieved in each tasks. In Python, retrieving the environment variable value is done as follow:
    ```
    import os # Retrieve environment variable can be done through os.environ

    MASTER_ADDR = os.environ["MASTER_ADDR"]
    ```


    **Running the tasks**

    `srun uv run "$@"`
    
    * The command `srun` creates tasks. The number of tasks is determined by the parameters `--ntasks` of our allocation. Here, we requested 4 tasks so the command will run 4 times in parallel tasks. These tasks run the command following `srun`, so each tasks will run `uv run "$@"`.

    * `uv run` is used to ease the environment set up for our tasks. For more information, read [our `uv` guide on portability](../../../userguides/python_uv).

    * `$@` transfers the parameters we gave while launching the script `job.sh`. In our case, we choose the script we want to run through `srun uv run`. Thus, when we launch `job.sh`, we use the command:
        * `sbatch job.sh main_jax.py` if we want to use the Jax example
        * `sbatch job.sh main_torch.py` if we want to use the Pytorch example.

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
        #print(f"{jax.local_devices()=}, {jax.devices()=}")

        # Compute all-reduce to compute the average across all processes.
        sum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
        if NODE_INDEX == 0 and RANK == 0:
            print(f"sum={sum[0]}")


    if __name__ == "__main__":
        main()
    ```

=== "main_torch.py"
    ```python
    import logging
    import os
    import subprocess

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

        sum = torch.clone(x)
        torch.distributed.reduce(
            sum, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if NODE_INDEX == 0 and RANK == 0: # The complete sum is done on the first tasks of the first node
            print(f"sum={sum[0]}")
        torch.distributed.destroy_process_group(group)


    if __name__ == "__main__":
        main()
    ```

??? info "In-depth script explaination on `main_***.py`"
    **Jax and Pytorch**

    This guide is based on two examples

    * [Jax](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) is a library for array-oriented numerical computation.

    * [Pytorch](https://pytorch.org/) is an open-source deep-learning library.


    **Environment variables**

    In each file, we retrieve the Slurm environment variables `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS` and `SLURM_NODEID`. Unlike the environment variables we defined previously (`MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE`), these environment variables are specific to each tasks. More SLURM common environment variables are listed in [the technical reference](../../../technical_reference/general_theory/slurm).

    ```python
    RANK = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    NODE_INDEX = int(os.environ["SLURM_NODEID"])
    ```

    === "What happens in Jax script"
        1. Initialize: this function is specific to Jax

        2. Create a value, different for each task

            The created value is based on the RANK, which is specific to each task

        3. Compute their sum [see Jax Lax parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators)
    
    === "What happens in Torch script"
        1. Initialize: in Torch, a group is defined

        2. Create a value, different for each task
            The created value is based on the RANK, which is specific to each task

        3. Compute their sum

    The final sum is printed from the first task of the first node (NODE_INDEX=0 and RANK=0). This is the task where all the `x` values have been collected. On the other tasks, the `sum` is a partial result.


=== "pyproject.toml"
    ```
    [project]
    name = "multitasks-demo"
    version = "0.1.0"
    description = "Using Jax and Torch to illustrate a multitask example"
    requires-python = ">=3.11,<3.14"
    dependencies = ["torch>=2.7.1", "jax[cuda12]>=0.5.3"]
    ```


??? info "In-depth explaination on `pyproject.toml`"
    `pyproject.toml` is a configuration file used by packaging tools (`uv` in our case) ([More info on `pyproject.toml` files](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)). The value of dependencies contains information about the libraries we are using in this example. `torch` is used while using the `main_torch.py` script, and `jax` while using the `main_jax.py` script. If you use only one of them, you can delete the unused library from the `pyproject.toml` file.

### Launching the example

1. Connect to the cluster

    ```bash
    ssh mila
    ```

2. Launch the job
    
    === "Launch Jax example"

        ```bash
        sbatch job.sh main_jax.py
        ```

    === "Launch Torch example"
        
        ```bash
        sbatch job.sh main_torch.py
        ```


3. (Optional) Check the job status

    ```bash
    squeue --me
    ```

4. Retrieve the results

    When the resources have been allocated and the script has run, an output file has been created: it is by default called `slurm-{JOB_ID}.out`, with `JOB_ID` being the ID of the job which has run.

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

    For each example, we can see that the ranks of the tasks (ie their `x` values) are respectively 0, 1, 2 and 3. Thus, their sum, retrieved on [Node 0 | Task 0], is 6.
