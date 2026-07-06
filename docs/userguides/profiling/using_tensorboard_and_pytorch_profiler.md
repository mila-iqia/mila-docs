---
title: Visualizing usage with Pytorch profiler and Tensorboard
description: >-
    This guide depicts a way to display usage of jobs run on the cluster
    by using the visualization toolkit Tensorboard alongside Pytorch profiler.
---

# Visualizing usage with Pytorch profiler and Tensorboard

This guide depicts a way to visualize metrics of jobs run on the cluster
by using the visualization toolkit [Tensorboard](https://www.tensorflow.org/tensorboard)
alongside [Pytorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Getting started with the Cluster__](../../../getting_started/)
    { .card }

    ---
    Get your Mila account, enable cluster access and MFA, then install `uv` and
    `milatools` to connect via SSH.


-   [:material-lightbulb-alert-outline:{ .lg .middle } __Understanding Slurm__](../../../userguides/slurm_guide/basics)
    { .card }

    ---
    Ask for a resource allocation and launch tasks on the cluster through an interactive job.


-   [:material-language-python:{ .lg .middle } __Managing Python Dependencies with `uv`__](../../../userguides/python_uv)
    { .card }

    ---
    Install uv, manage project dependencies, run reproducible Slurm jobs, and run
    standalone scripts.


-   [:material-magnify:{ .lg .middle } __Identifying GPU waste__](../../../userguides/gpu_efficiency.md)
    { .card }

    ---
    Introduce the notion of profiling.

&nbsp;

</div>

## What this guide covers
* Introduce Tensorboard and Pytorch profiler
* Launch Tensorboard alongside jobs on the cluster

---

## Description of the process

TensorBoard reads profiling data from a directory that you specify when launching it.
Visualizing a job's performance with TensorBoard involves two steps:

1. **Recording profiling data:** PyTorch Profiler writes trace files to the directory during the job's execution.
2. **Viewing the metrics:** TensorBoard is launched pointing to that directory, either while the job is still running or after it has finished.

## Write a code example

!!! info
    This guide is based on the following guides from the Pytorch documentation:

    * [How to use TensorBoard with PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
    * [Pytorch profiler with Tensorboard](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/50d8e1e8ecc86503893ab8f9f52932ba/tensorboard_profiler_tutorial.ipynb)

    You can refer to them for more details.


### Base code
The following code is an example of training a model with Pytorch:
```python
import torch

# Linear regression training example
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
```

### Writing the job usage

To write the metrics, we use `profile` from the `torch.profiler` 
library. The template of writing metrics is as follows:

```python
import os

# Import Pytorch profiler
from torch.profiler

# Define in which folder we want the results to be stored
# (if the folder does not exist, it is created)
scratch_path = os.environ.get("SCRATCH")
job_id = os.environ.get("SLURM_JOB_ID")
folder_name = f"{scratch_path}$/runs/{job_id}_profiling"

# Initialize the profiler
#   - schedule:
#   - on_trace_ready: 
#   - record_shapes: 
#   - with_stack: 
profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(folder_name),
        record_shapes=True,
        with_stack=True)

# Start the profiler
profiler.start()


# Train the model
[...]

# Write the metrics while training the model
profiler.step()

[...]


# Stop the profiler when you do not need it anymore
profiler.stop()
```

This results in the following code (ready for use):
```python
import os
import torch
# Import Pytorch profiler
import torch.profiler


# Linear regression training example
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Define in which folder we want the results to be stored
# (if the folder does not exist, it is created)
scratch_path = os.environ.get("SCRATCH")
job_id = os.environ.get("SLURM_JOB_ID")
folder_name = f"{scratch_path}$/runs/{job_id}_profiling"

profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(folder_name),
            record_shapes=True,
            with_stack=True)

# Start the profiler
profiler.start()

# While the model is training
def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Write the metrics while training the model
        profiler.step()

# Train the model
train_model(10)

# Stop the profiler when you do not need it anymore
profiler.stop()
```


## Try this example locally

Launching the example locally is done through the following steps:

1. Write the experiment code
2. Set up the environment
3. Launch the experiment
4. Launch Tensorboard
5. Access Tensorboard visualization

### Write the experiment code
We use the code explained in [the previous section](#write-a-code-example).

=== "experiment.py"
    ```python
    import os
    import torch
    # Import Pytorch profiler
    import torch.profiler


    # Linear regression training example
    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    # Define in which folder we want the results to be stored
    # (if the folder does not exist, it is created)
    scratch_path = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    folder_name = f"{scratch_path}$/runs/{job_id}_profiling"

    profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(folder_name),
                record_shapes=True,
                with_stack=True)

    # Start the profiler
    profiler.start()

    # While the model is training
    def train_model(iter):
        for epoch in range(iter):
            y1 = model(x)
            loss = criterion(y1, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Write the metrics while training the model
            profiler.step()

    # Train the model
    train_model(10)

    # Stop the profiler when you do not need it anymore
    profiler.stop()
    ```

### Set up the environment
To easily set the environment for this example, we use `uv`. If it has already be done,
you can skip this section and go to [Launch the experiment](#launch-the-experiment).

1. [Optional if already done] The first step is to install `uv` : this is explained in [this section](../../../userguides/python_uv/#install-uv).

2. We then initialize the project. This create a `pyproject.toml` file.
```
uv init
```

3. In this example, we use `torch`, `torch-tb-profiler` and `tensorboard`, so we add them to the environment configuration:
```
uv add torch
uv add torchvision
uv add torch-tb-profiler
uv add tensorboard
```

### Launch the experiment
Launching the experiment is done through the command:
```
uv run python experiment.py
```

The folder `runs/experiment1` has been created.


### Launch Tensorboard
Tensorboard can be launched whether the job is running or has ended, this is done through the command:
```
uv run tensorboard --logdir=runs --port=6006
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
</div>

### Access Tensorboard visualization
You can access Tensorboard interface through localhost, by using the port defined in the previous step (here `6006`). To this end, open a navigator and enter `127.0.0.1:6006` in the address bar.

The following dashboard appears:
![Tensorboard dashboard](../../../_static/images/tensorboard_dashboard_usage_one_experiment.png)


## Launch this example on the cluster

### When the job has run
One of the main challenge on the cluster is to retrieve the data. If we want to visualize
them after the experiment has run, we simply can copy the directory where the metrics are
logged (`runs` in the previous section).

### While the job is running
However, if the metrics have to be monitored "live", we are following these steps:

1. Connect to the cluster
2. Set up the project for the cluster
3. Launch the experiment
4. Launch Tensorboard
5. Access Tensorboard through port forwarding.

#### Connect to the cluster
This step is configured and explained in the [Getting started guide](../../../getting_started/).

Here, we arbitrary choose to use the Mila cluster.

```
ssh mila
```

We are now connected on one of the login node.


#### Set up the project for the cluster
This could be done through `uv init` and `uv add <dependencies>` such as presented in [Try this example locally - Set up the environment](#set-up-the-environment). However, this is the opportunity to underline the portability of `uv`: we define the environment by copying the `pyproject.toml` file.

Hence, we copy (or write) the following files on the login node:

=== "experiment.py"
    ```python
    import os
    import torch
    # Import Pytorch profiler
    import torch.profiler


    # Linear regression training example
    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    # Define in which folder we want the results to be stored
    # (if the folder does not exist, it is created)
    scratch_path = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    folder_name = f"{scratch_path}$/runs/{job_id}_profiling"

    profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(folder_name),
                record_shapes=True,
                with_stack=True)

    # Start the profiler
    profiler.start()

    # While the model is training
    def train_model(iter):
        for epoch in range(iter):
            y1 = model(x)
            loss = criterion(y1, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Write the metrics while training the model
            profiler.step()

    # Train the model
    train_model(10)

    # Stop the profiler when you do not need it anymore
    profiler.stop()
    ```

=== "pyproject.toml"
    ```
    [project]
    name = "test2"
    version = "0.1.0"
    description = "Add your description here"
    readme = "README.md"
    requires-python = ">=3.14"
    dependencies = [
        "tensorboard>=2.20.0",
        "torch>=2.12.0",
        "torch-tb-profiler>=0.4.3",
        "torchvision>=0.27.0",
    ]
    ```

#### Launch the experiment
Launching an experiment on the cluster is explained in the [Launching jobs guide](../../../slurm_guide/). You have two ways to do this:

* an interactive job
* a batch job.

Here, we use the batch job option. Thus, we create a `job.sh` file:

=== "job.sh"
    ```bash
    #!/bin/bash
    #SBATCH --ntasks=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1
    #SBATCH --gpus-per-task=rtx8000:1
    #SBATCH --time=00:03:00

    # Exit on error
    set -e

    # Echo time and hostname into log
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"

    # Execute Python script
    # Use `uv run --offline` on clusters without internet access on compute nodes.
    srun uv run python experiment.py
    ```

Finally, we launch the job through the command line:

```
sbatch job.sh
```

#### Launch Tensorboard

!!! warning "Do not launch Tensorboard on the login node"
    Login nodes exist for light interactions, mainly interacting with Slurm.
    Launching Tensorboard on a login node will consume too much memory, and
    cause problems for other people too.

    Thus, Tensorboard must be launched on compute node if launched on the cluster.

As we want to launch Tensorboard on a compute node, we retrieve the name of the node
on which we are running. This could be done through the following Slurm environment
variables:

| Slurm environment variable | Description |
| -------------------------- | ----------- |
| `$SLURM_NODELIST`          | Contains the full list of compute nodes allocated to the job. |
| `$SLURMD_NODENAME`         | Identifies the specific, individual node name that is currently executing the script step. |


For this example, we do it manually by calling `squeue --me` to check our job status, our job ID and the node name. In the following case:

* the job ID is `9858178`
* the node is `cn-a007`.

```
squeue --me
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
JOBID     USER    PARTITION           NAME  ST START_TIME             TIME NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) COMMENT
 9858178 <USER>         long         job.sh   R 2026-06-17T14:09       1:50     1    1        N/A      2G cn-a007 (None) (null)
```
</div>


In order to connect to the compute node, we can use:

```
srun -s --pty --jobid=9858178 /bin/bash
```
(Note to adapt this command to your job ID.)

Once on the compute node (the node name should appear in your terminal), we can launch
Tensorboard through:

```
tensorboard --logdir=runs --port=16123
```

#### Access Tensorboard through port forwarding

Finally, in order to access the dashboard on our localhost, we have to tunnel information
from the compute node to our local machine.

Open a terminal on your local machine and use the following command (replacing `cn-a007` by the node name retrieved in the previous step):

```
ssh -L 16006:localhost:16123 cn-a007.server.mila.quebec
```

From now on, you should be able to access the Tensorboard dashboard by opening your
navigator and enter the address `127.0.0.1:16006`.

---

## Key concepts

SSH port forwarding
:   Also called "SSH tunneling", it is an operation where a machine listens on a specific port, and transfers it to a (potentially other port) on another machine. [More info here](https://www.ssh.com/academy/ssh/tunneling)
