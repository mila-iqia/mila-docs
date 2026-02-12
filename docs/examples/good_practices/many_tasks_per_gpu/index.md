# Launch many tasks on the same GPU


If you want to use a powerful GPU efficiently, you can run many tasks on same GPU
using a combination of `sbatch` arguments. In your `sbatch` script:

- Specify only 1 GPU to use, e.g. with `--gres=gpu:rtx8000:1`
- Specify number of tasks to run on the selected GPU with `--ntasks-per-gpu=N`
- Launch your job using `srun main.py` instead of just `main.py`.

`srun` will then launch `main.py` script `N` times.
Each task will receive specific environment variables, such as `SLURM_PROCID`,
which you can then use to parameterize the script execution.

**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* [PyTorch setup](../../frameworks/pytorch_setup/index.md)

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/many_tasks_per_gpu)

**job.sh**

```diff
--8<-- "docs/examples/good_practices/many_tasks_per_gpu/job.sh.diff"
```

**main.py**

```diff
--8<-- "docs/examples/good_practices/many_tasks_per_gpu/main.py.diff"
```

## Running this example

You can launch this example with sbatch:

```bash
 $ sbatch job.sh
```
