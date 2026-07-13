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
from pathlib import Path

# Import Pytorch profiler
import torch.profiler

# Define in which folder we want the results to be stored
SCRATCH = Path(os.environ.get("SCRATCH", "fake_scratch"))
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")

logs_dir = SCRATCH / "logs" / SLURM_JOB_ID
logs_dir.mkdir(parents=True, exist_ok=True)

# Initialize the profiler
#   - schedule:
#   - on_trace_ready: 
#   - record_shapes: 
#   - with_stack: 
profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
    record_shapes=True,
    with_stack=True,
)

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

### Ready-for-use code
Putting all of this together, here is an example you can run directly:

=== "experiment.py"
    ```python
    import os
    from pathlib import Path
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
    SCRATCH = Path(os.environ.get("SCRATCH", "fake_scratch"))
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")

    logs_dir = SCRATCH / "logs" / SLURM_JOB_ID
    logs_dir.mkdir(parents=True, exists_ok=True)

    
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
        record_shapes=True,
        with_stack=True,
    )

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
We use the code explained in [the previous section](#ready-for-use-code).

### Set up the environment
The environment is described in the following file. Copying it as `pyproject.toml` would make available all the prerequisites
while running the `uv` command.

=== "pyproject.toml"
    ```
    [project]
    name = "my_project"
    version = "0.1.0"
    description = "Add your description here"
    readme = "README.md"
    requires-python = ">=3.14"
    dependencies = [
        "tensorboard>=2.21.0",
        "torch>=2.12.1",
        "torch-tb-profiler>=0.4.3",
        "torchvision>=0.27.1",
    ]
    ```

### Launch the experiment
Once the two files (`experiment.py` and `pyproject.toml`) have been written in your environment, you can
launch the experiment through the following command:
```console
uv run python experiment.py
```

The folder `$SCRATCH/logs/$SLURM_JOB_ID` has been created.


### Launch Tensorboard
Tensorboard can be launched whether the job is running or has ended, this is done through the command:
```console
uv run tensorboard --logdir=runs
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

However, if the metrics have to be monitored "live", we are following these steps:

1. Connect to the cluster
2. Set up the project for the cluster
3. Launch the experiment
4. Launch Tensorboard
5. Access Tensorboard.

### Connect to the cluster
This step is configured and explained in the [Getting started guide](../../../getting_started/).

Here, we choose to use the Mila cluster.

```console
ssh mila
```

We are now connected on one of the login node.


### Set up the project for the cluster
This could be done through `uv init` and `uv add <dependencies>` such as presented in [Try this example locally - Set up the environment](#set-up-the-environment). However, this is the opportunity to underline the portability of `uv`: we define the environment by copying the `pyproject.toml` file.

Hence, we copy (or write) the following files on the login node:

=== "experiment.py"
    ```python
    import os
    from pathlib import Path
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
    SCRATCH = Path(os.environ.get("SCRATCH", "fake_scratch"))
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")
    logs_dir = SCRATCH / "logs" / SLURM_JOB_ID
    logs_dir.mkdir(parents=True, exists_ok=True)

    
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
        record_shapes=True,
        with_stack=True,
    )

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

### Launch the experiment
Launching an experiment on the cluster is explained in the [Launching jobs guide](../../../slurm_guide/). You have two ways to do this:

* an interactive job with `salloc`
* a batch job with `sbatch`.

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

```console
sbatch job.sh
```

### Launch Tensorboard

!!! warning "Do not launch Tensorboard on the login node"
    Login nodes exist for light interactions, mainly interacting with Slurm.
    Launching Tensorboard on a login node will consume too much memory, and
    cause problems for other people too.

    Thus, Tensorboard must be launched on compute node if launched on the cluster.



Finally, we want to launch Tensorboard and access the dashboard on our localhost. This could be done with two methods:

* by using `ssh mila-cpu` (**Recommended**)
* through port forwarding.


=== "`ssh mila-cpu` (**Recommended**)"

    You can use this methode if you have already launched `mila code` (cf [Getting started](../../../getting_started/#install-milatools)):

    1. Launch VSCode

    2. Open the "Remote Explorer" in the left menu of VSCode

        ![VSCode profiling 1](../../../_static/images/vscode_profiling_1.png)
        
    3. Find `mila-cpu` in the list and click on "Connect in Current Window" or "Connect in New Window"

        ![VSCode profiling 2](../../../_static/images/vscode_profiling_2.png)

        Note: the setup could take a moment. Meanwhile, the following message appears:
        
        ![VSCode profiling 3](../../../_static/images/vscode_profiling_3.png)

    4. Open the Terminal and write the command:
        
        ```console
        uvx tensorboard --logdir $SCRATCH/logs/$SLURM_JOB_ID
        ```
    
    5. Access the Tensorboard dashboard by opening your browser and enter the address [http://localhost:6006](http://localhost:6006).


=== "Port forwarding"

    !!! warning "Performance"
        Launching Tensorboard directly on the requested nodes would affect the
        job's resources.

    Another option to access the dashboard on our localhost is to tunnel information
    from the compute node to our local machine.

    1. Launch Tensoboard **on a compute node** by using the command:

        ```console
        uvx tensorboard --logdir $SCRATCH/logs/$SLURM_JOB_ID
        ```

    2. Open a terminal on your local machine and use the following command (replacing `cn-f003` by the node name where Tensorboard has been launched):

        ```console
        ssh -L 6006:localhost:6006 cn-f003.server.mila.quebec
        ```

    3. Access the Tensorboard dashboard by opening your browser and enter the address [http://localhost:6006](http://localhost:6006).

    !!! tip "Changing ports"
        Here, `6006` is the default port for Tensorboard. It can be changed when Tensorboard is launched, by using the parameter `--port`.
        In this case, the ports in the port forwarding should be changed too.

---

## Key concepts

SSH port forwarding
:   Also called "SSH tunneling", it is an operation where a machine listens on a specific port, and transfers it to a (potentially other port) on another machine. [More info here](https://www.ssh.com/academy/ssh/tunneling)
