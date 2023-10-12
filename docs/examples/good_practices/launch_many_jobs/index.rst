.. _launch_many_jobs:

Launch many jobs from same shell script
=======================================

Sometimes you may want to run same script by changing just few values.

Instead of "comment line, change value, and re-run" cycle, you can
parameterize the script and then call it multiple times with different parameters.

Here we provide an example, with Python script accepting parameters, and
bash script just receiving and passing parameters to Python script.

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
