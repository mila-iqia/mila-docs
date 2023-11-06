.. _launch_many_jobs:

Launch many jobs from same shell script
=======================================

Sometimes you may want to run the same job with different arguments. For example, you may want to launch an experiment using a few different values for a given parameter.

The naive way to do this would be to create multiple sbatch scripts, each with a different value for that parameter.
Another might be to use a single sbatch script with multiple lines, each with a different parameter value, and to then uncomment a given line before submitting the job, then commenting and uncommenting a different line before submitting another job, etc.

This example shows a  practical solution to this problem, allowing you to parameterize a job's sbatch script, and pass different values directly from the command-line when submitting the job.

In this example, our job script is a slightly modified version of the Python script from the single-GPU example, with a bit of code added so that it is able to take in values from the command-line.
The sbatch script uses the ``$@`` bash directive to pass the command-line arguments to the python script. This makes it very easy to submit multiple jobs, each with different values!

The next examples will then build on top of this one to illustrate good practices related to launching lots of jobs for hyper-parameter sweeps:

* Using SLURM Job Arrays for Hyper-Parameter Sweeps (coming soon!)
* :ref:`Running more effective Hyper-Parameter Sweeps with Orion <hpo_with_orion>`


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/launch_many_jobs>`_

**job.sh**

.. literalinclude:: job.sh.diff
    :language: diff


**main.py**

.. literalinclude:: main.py.diff
    :language: diff


**Running this example**

This assumes you already created a conda environment named "pytorch" as in
Pytorch example:

* :ref:`pytorch_setup`

Exit the interactive job once the environment has been created.
You can then launch many jobs using same script with various args.

.. code-block:: bash

    $ sbatch job.sh --learning-rate 0.1
    $ sbatch job.sh --learning-rate 0.5
    $ sbatch job.sh --weight-decay 1e-3
