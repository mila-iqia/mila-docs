.. _single_gpu_job:

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

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/slurm_job_arrays>`_


**main.py**

.. literalinclude:: main.py.diff
    :language: diff


**Running this example**

This assumes you already created a conda environment named "pytorch" as in
Pytorch example:

* :ref:`pytorch_setup`

Exit the interactive job once the environment has been created.
You can then launch a job array using ``sbatch`` argument ``--array``.

.. code-block:: bash

    $ sbatch --array=1-5 job.sh


In this example, 5 jobs will be launched with indices (therefore, values of ``SLURM_ARRAY_TASK_ID``) from 1 to 5.
