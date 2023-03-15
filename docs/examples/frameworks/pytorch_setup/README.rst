PyTorch Setup
===================

.. IDEA: Add a link to all the sections of the documentation that have to
.. absolutely have been read before this tutorial.

**Prerequisites**: (Make sure to read the following before using this example!)

* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`Conda`


**job.sh**


.. literalinclude:: /examples/frameworks/pytorch_setup/job.sh
    :language: bash


**main.py**


.. literalinclude:: /examples/frameworks/pytorch_setup/main.py
    :language: python


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
