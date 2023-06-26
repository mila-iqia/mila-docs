Hyperparameter Optimization with Oríon
======================================

There are frameworks that allow to do hyperparameter optimization, like
`wandb <https://wandb.ai/>`_,
and `Oríon <https://orion.readthedocs.io/en/stable/index.html>`_.
Here we provide an example for Oríon, the HPO framework developped at Mila.

**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`

The full documentation for Oríon is available `on Oríon's ReadTheDocs page
<https://orion.readthedocs.io/en/stable/index.html>`_.


**job.sh**

.. literalinclude:: examples/hpo/hpo_with_orion/job.sh.diff
    :language: diff

.. .. literalinclude:: examples/hpo/hpo_with_orion/job.sh
..     :language: bash


**main.py**

.. literalinclude:: examples/hpo/hpo_with_orion/main.py.diff
    :language: diff

.. .. literalinclude:: examples/hpo/hpo_with_orion/main.py
..     :language: python


**Running this example**

This assumes you already created a conda environment named "pytorch" as in
Pytorch example:

* :ref:`pytorch_setup`

Oríon must be installed inside the "pytorch" environment using following command:

.. code-block:: bash

    pip install orion[profet]

Exit the interactive job once the environment has been created and Oríon installed.
You can then launch the example:

.. code-block:: bash

    $ sbatch job.sh

To get more information about the optimization run, activate "pytorch" environment
and run ``orion info`` with the experiment name:

.. code-block:: bash

    $ conda activate pytorch
    $ orion info -n orion-example
