# Multi-Node / Multi-GPU ImageNet Training


Prerequisites:

* [PyTorch Setup](../../frameworks/pytorch_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)
* [Multi-GPU](../../distributed/multi_gpu/index.md)
* [Multi-node](../../distributed/multi_node/index.md)
* [Checkpointing](../../good_practices/checkpointing/index.md)
* [Launch many jobs](../../good_practices/launch_many_jobs/index.md)

Other interesting resources:

* [https://sebarnold.net/dist_blog/](https://sebarnold.net/dist_blog/)
* [Multi-node PyTorch distributed training guide](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)


Click here to see [the source code for this example](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/advanced/imagenet)


This is an advanced and quite lengthy example. We recommend viewing the files directly
on GitHub to get the best experience.


**pyproject.toml**

This is the configuration file for UV, which manages the dependencies for this project.

```toml
--8<-- "docs/examples/advanced/imagenet/pyproject.toml"
```

**safe_sbatch**

This job script uses the `safe_sbatch` submission script to submit a job at the current git state.
This practice is recommended to ensure reproducibility, and to prevent changes in the python files between when the job
is submitted and when it starts to affect the results.

Unlike the script passed to sbatch, which is copied and saved with the job in SLURM (and reused when resuming a job),
the python files are not saved.

```bash
--8<-- "docs/examples/advanced/imagenet/safe_sbatch"
```

**job.sh**

This file uses a `code_checkpointing.sh` utility script.
For now, to keep this already very heavy example a bit lighter,
we do not include it here, but you can find it in the GitHub repository [here](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/advanced/imagenet)


```bash
--8<-- "docs/examples/advanced/imagenet/job.sh"
```

**prepare_data.py**

This script downloads and prepares the ImageNet dataset.
You need to run it once before running the main training script.

```python
--8<-- "docs/examples/advanced/imagenet/prepare_data.py"
```

**main.py**

```python
--8<-- "docs/examples/advanced/imagenet/main.py"
```

## Running this example

You can submit this as a batch job with sbatch, or you can run it in an interactive job with `srun`:

```bash
$ sbatch job.sh
```

or, for example in an interactive job:

```bash
$ ssh mila 'git clone https://github.com/mila-iqia/mila-docs'

# Get an interactive job. You can use as many nodes or gpus, in whatever configuration you wish.
# Here we choose to use between 1 and 2 nodes, with 4 GPUs distributed in any between the two nodes. (could be 4-0, 3-1, 2-2, etc.)
$ ssh -tt mila salloc --nodes=1-2 --ntasks=4 --gpus-per-task=l40s:1 --cpus-per-task=4 --mem=32G --tmp=200G --time=02:59:00 --partition=short-unkillable
salloc: Granted job allocation 7782523
salloc: Waiting for resource configuration
salloc: Nodes cn-l[023,054] are ready for job

# Run the dataset preparation on each node:
$ cd mila-docs/docs/examples/advanced/imagenet
$ srun --ntasks-per-node=1 uv run python prepare_data.py

# Run the training script on each gpu on each node
# NOTE: this only works in an interactive terminal with salloc! For the VsCode integrated terminal, see below.
$ srun uv run python prepare_data.py
```

To open this example with VsCode:

```bash
$ mila code mila-docs/docs/examples/advanced/imagenet --alloc --ntasks=4 --gpus-per-task=l40s:1 --mem=32G --tmp=200G --time=02:59:00 --partition=short-unkillable
# Or, if you are already in a terminal in an interactive job:
$ mila code mila-docs/docs/examples/advanced/imagenet --job 7782523
```

Then, in the vscode terminal, you will have to explicitly list out the number of nodes and tasks to use, since those
can't be inferred from the SLURM environment variables (which are not present, since you are SSH-ing into the compute node).

```bash
# If your job has 2 nodes, for example:
$ srun --ntasks-per-node=1 --nodes=2 uv run python prepare_data.py
# Launch the training script on each gpu on each node
$ srun --ntasks=4 --nodes=2 uv run python main.py
```

