Hyper-Parameter Optimization using Oríon
========================================

There are frameworks that allow to do hyper-parameters optimization, like
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

.. literalinclude:: examples/frameworks/orion_setup/job.sh.diff
    :language: diff

.. .. literalinclude:: examples/frameworks/orion_setup/job.sh
..     :language: bash


**main.py**

.. literalinclude:: examples/frameworks/orion_setup/main.py.diff
    :language: diff

.. .. literalinclude:: examples/frameworks/orion_setup/main.py
..     :language: python


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
