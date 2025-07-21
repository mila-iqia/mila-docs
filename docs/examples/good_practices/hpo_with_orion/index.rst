.. _hpo_with_orion:

Hyperparameter Optimization with Oríon
======================================

There are frameworks that allow to do hyperparameter optimization, like
`wandb <https://wandb.ai/>`_,
and `Oríon <https://orion.readthedocs.io/en/stable/index.html>`_.
Here we provide an example for Oríon, the HPO framework developped at Mila.

**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :doc:`/examples/frameworks/pytorch_setup/index`

The full documentation for Oríon is available `on Oríon's ReadTheDocs page
<https://orion.readthedocs.io/en/stable/index.html>`_.


The full source code for this example is available on `the mila-docs GitHub repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/hpo_with_orion>`_


**job.sh**

.. literalinclude:: job.sh.diff
    :language: diff

**pyproject.toml**

.. literalinclude:: pyproject.toml
    :language: toml

**main.py**

.. literalinclude:: main.py.diff
    :language: diff

**Running this example**

This assumes you already created a conda environment named "pytorch" as in
Pytorch example:

* :ref:`pytorch_setup`

Oríon must be installed inside the "pytorch" environment using following command:

.. code-block:: bash

    pip install orion

Exit the interactive job once the environment has been created and Oríon installed.
You can then launch the example:

.. code-block:: bash

    $ sbatch job.sh

To get more information about the optimization run, activate "pytorch" environment
and run ``orion info`` with the experiment name:

.. code-block:: bash

    $ conda activate pytorch
    $ orion info -n orion-example

You can also generate a plot to visualize the optimization run. For example:

.. code-block:: bash

    $ orion plot regret -n orion-example

For more complex and useful plots, see `Oríon documentation
<https://orion.readthedocs.io/en/stable/auto_examples/plot_4_partial_dependencies.html>`_.
