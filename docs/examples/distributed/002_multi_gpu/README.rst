002 - Multi-GPU Job
====================


Prerequisites:

* :ref:`001 - PyTorch Setup`
* :ref:`001 - Single GPU Job`

Other interesting resources:
* `https://sebarnold.net/dist_blog/`
* `https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide`

.. literalinclude:: /examples/distributed/002-multi-gpu/job.sh
    :language: bash

.. literalinclude:: /examples/distributed/002-multi-gpu/main.py
    :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
