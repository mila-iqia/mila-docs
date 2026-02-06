<a id="launch_many_jobs"></a>

### Launch many jobs from same shell script


Sometimes you may want to run the same job with different arguments.
For example, you may want to launch an experiment using a few different learning rates.
This example shows an easy way to do this.


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* [PyTorch Setup](../../frameworks/pytorch_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)


**job.sh**

Compared to the [single GPU job](../../distributed/single_gpu/index.md) example, here we use the `$@` bash directive
to pass command-line arguments down to the Python script.

This makes it very easy to submit multiple jobs, each with different values!

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/launch_many_jobs)

```diff
--8<-- "docs/examples/good_practices/launch_many_jobs/job.sh.diff"
```

**Running this example**

You can run this example just like the [single GPU job](../../distributed/single_gpu/index.md) example, but you can now
also pass command-line arguments directly when submitting the job with `sbatch`!

For example:

```bash
 $ sbatch job.sh --learning-rate 0.1
 $ sbatch job.sh --learning-rate 0.5
 $ sbatch job.sh --weight-decay 1e-3
```


##### Next steps


These next examples build on top of this one and show how to properly launch lots of jobs for hyper-parameter sweeps:
* [Using SLURM Job Arrays to launch lots of jobs](../slurm_job_arrays/index.md)
* [Running more effective Hyper-Parameter Sweeps with Orion](../hpo_with_orion/index.md)
