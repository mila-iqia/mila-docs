003 - Multi-Node-Multi-GPU (DDP) Job
=====================================


Prerequisites:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`
* :ref:`002 - Multi-GPU Job`

Other interesting resources:
* `https://sebarnold.net/dist_blog/`
* `https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide`

.. literalinclude:: /examples/distributed/003_multi_node/job.sh
    :language: bash

.. literalinclude:: /examples/distributed/003_multi_node/main.py
    :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
