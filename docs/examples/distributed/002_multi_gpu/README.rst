002 - Multi-GPU Job
====================


Prerequisites:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

Other interesting resources:
* `https://sebarnold.net/dist_blog/`
* `https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide`

.. literalinclude:: /examples/distributed/002_multi_gpu/job.sh
    :language: bash

.. literalinclude:: /examples/distributed/002_multi_gpu/main.py
    :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
