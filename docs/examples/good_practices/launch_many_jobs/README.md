<!-- NOTE: This file is auto-generated from examples/good_practices/launch_many_jobs/index.md
     This is done so this file can be easily viewed from the GitHub UI.
     DO NOT EDIT -->

<a id="launch_many_jobs"></a>

### Launch many jobs from same shell script


Sometimes you may want to run the same job with different arguments.
For example, you may want to launch an experiment using a few different learning rates.
This example shows an easy way to do this.


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* [PyTorch Setup](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup)
* [Single GPU](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu)


**job.sh**

Compared to the [single GPU job](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu) example, here we use the `$@` bash directive
to pass command-line arguments down to the Python script.

This makes it very easy to submit multiple jobs, each with different values!

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/launch_many_jobs)

```diff
 # distributed/single_gpu/job.sh -> good_practices/launch_many_jobs/job.sh
 #!/bin/bash
 #SBATCH --ntasks=1
 #SBATCH --ntasks-per-node=1
 #SBATCH --cpus-per-task=4
 #SBATCH --gpus-per-task=l40s:1
 #SBATCH --mem-per-gpu=16G
 #SBATCH --time=00:15:00
 
 # Exit on error
 set -e
 
 # Echo time and hostname into log
 echo "Date:     $(date)"
 echo "Hostname: $(hostname)"
 
 # To make your code as much reproducible as possible with
 # `torch.use_deterministic_algorithms(True)`, uncomment the following block:
 ## === Reproducibility ===
 ## Be warned that this can make your code slower. See
 ## https://pytorch.org/docs/stable/notes/randomness.html#cublas-and-cudnn-deterministic-operations
 ## for more details.
 # export CUBLAS_WORKSPACE_CONFIG=:4096:8
 ## === Reproducibility (END) ===
 
 # Stage dataset into $SLURM_TMPDIR
 mkdir -p $SLURM_TMPDIR/data
 cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
 # General-purpose alternatives combining copy and unpack:
 #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
 #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/
 
 # Execute Python script
 # Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
 # Using the `--locked` option can help make your experiments easier to reproduce (it forces
 # your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
-srun uv run python main.py
+srun uv run python main.py "$@"

```

**Running this example**

You can run this example just like the [single GPU job](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu) example, but you can now
also pass command-line arguments directly when submitting the job with `sbatch`!

For example:

```bash
 $ sbatch job.sh --learning-rate 0.1
 $ sbatch job.sh --learning-rate 0.5
 $ sbatch job.sh --weight-decay 1e-3
```


##### Next steps


These next examples build on top of this one and show how to properly launch lots of jobs for hyper-parameter sweeps:
* [Using SLURM Job Arrays to launch lots of jobs](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/slurm_job_arrays)
* [Running more effective Hyper-Parameter Sweeps with Orion](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/hpo_with_orion)
