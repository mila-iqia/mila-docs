.. _many_tasks_per_gpu:

Launch many tasks on same GPU
=============================

If you want to use a powerful GPU efficiently, you can run many tasks on same GPU
using a combination of ``sbatch`` arguments. In your ``sbatch`` script:

- Specify only 1 GPU to use, e.g. with ``--gres=gpu:rtx8000:1``
- Specify number of tasks to run on the selected GPU with ``--ntasks-per-gpu=N``
- Launch your job using ``srun main.py`` instead of just ``main.py``.

``srun`` will then launch ``main.py`` script ``N`` times.
Each task will receive specific environment variables, such as ``SLURM_PROCID``,
which you can then use to parameterize the script execution.

**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/many_tasks_per_gpu>`_

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

Exit the interactive job once the environment has been created and Oríon installed.
You can then launch the example:

.. code-block:: bash

    $ sbatch job.sh
