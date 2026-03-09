.. _slurm_job_arrays:

Launch many jobs using SLURM job arrays
=======================================

Sometimes you may want to run many tasks by changing just a single parameter.

One way to do that is to use SLURM job arrays, which consists of launching an array of jobs using the same script.
Each job will run with a specific environment variable called ``SLURM_ARRAY_TASK_ID``, containing the job index value inside job array.
You can then slightly modify your script to choose appropriate parameter based on this variable.

You can find more info about job arrays in the `SLURM official documentation page <https://slurm.schedmd.com/job_array.html>`_.


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`
* :doc:`/examples/good_practices/checkpointing/index`
* :doc:`/examples/good_practices/many_tasks_per_gpu/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/slurm_job_arrays>`_


**main.py**

.. literalinclude:: main.py.diff
    :language: diff


**Running this example**

You can then launch a job array using ``sbatch`` with the ``--array`` argument, for example:

.. code-block:: bash

    $ sbatch --array=1-5 job.sh

In this example, 5 jobs will be launched with indices (therefore, values of ``SLURM_ARRAY_TASK_ID``) from 1 to 5.

Even better, you can combine job arrays with the many tasks per GPU (job packing) technique
explained in :doc:`/examples/good_practices/many_tasks_per_gpu/index`!
For example, this command will launch 10 jobs (10 sets of hyper-parameters), each using 5 tasks to
run 5 different initializations on the same GPU.

.. code-block:: bash

    $ sbatch --array=1-10 --ntasks-per-gpu=5 --gpus=1 --cpus-per-task=1 job.sh


Note that each task requires at least one CPU, so you may need to adjust the cpu count in your job in order to scale up ``--ntasks-per-gpu``.
Here with ``--cpu-per-task=1``, this will scale nicely.
