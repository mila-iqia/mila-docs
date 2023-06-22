Orion example on single GPU Job
===============================


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`

The full documentation for Oríon is available `here
<https://orion.readthedocs.io/en/stable/index.html>`_


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
