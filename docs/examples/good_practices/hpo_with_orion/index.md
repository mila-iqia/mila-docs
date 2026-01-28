### Hyperparameter Optimization with Oríon

There are different frameworks that allow you to do hyperparameter optimization, for example
[wandb ](https://wandb.ai/), [hydra ](https://hydra.cc/), [Ray Tune ](https://docs.ray.io/en/latest/tune/index.html)
and [Oríon ](https://orion.readthedocs.io/en/stable/index.html).
Here we provide an example for Oríon, the HPO framework developped at Mila.

[Orion ](https://orion.readthedocs.io/en/stable/?badge=stable)
is an asynchronous framework for black-box function optimization developped at Mila.

Its purpose is to serve as a meta-optimizer for machine learning models and training,
as well as a flexible experimentation platform for large scale asynchronous optimization procedures.

**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`

The full documentation for Oríon is available [on Oríon's ReadTheDocs page
](https://orion.readthedocs.io/en/stable/index.html).

The full source code for this example is available on [the mila-docs GitHub repository.
](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/hpo_with_orion)

Hyperparameter optimization is very easy to parallelize as each trial (unique set of hyperparameters) are
independant of each other. The easiest way is to launch as many jobs as possible each trying a different
set of hyperparameters and reporting their results back to a synchronized location (database).

**job.sh**

The easiest way to run an hyperparameter search on the cluster is simply to use a job array,
which will launch the same job n times. Your HPO library will generate different parameters to try
for each job.

Orion saves all the results of its optimization process in a database,
by default it is using a local database on a shared filesystem named `pickleddb`.
You will need to specify its location and the name of your experiment.
Optionally you can configure workers which will run in parallel to maximize resource usage.

**pyproject.toml**

This doesn't change much, the only difference is that we add the Orion dependency.

**main.py**

Here we only really add the reporting of the objective to Orion at the end of the main function.

**Running this example**

In the example below we use 10 jobs each with 5 CPU cores and one GPU.
Each job will run 5 tasks in parallel on the same GPU to maximize its utilization.
This means there could be 50 sets of hyperparameters being worked on in parallel across 10 GPUs.

```bash

    $ sbatch --array=1-10 --ntasks-per-gpu=5 --gpus=1 --cpus-per-task=1 job.sh

To get more information about the optimization run, use `orion info` with the experiment name:

```bash

    $ uv run orion info -n orion-example

You can also generate a plot to visualize the optimization run. For example:

```bash

    $ uv run orion plot regret -n orion-example

For more complex and useful plots, see [Oríon documentation
](https://orion.readthedocs.io/en/stable/auto_examples/plot_4_partial_dependencies.html).