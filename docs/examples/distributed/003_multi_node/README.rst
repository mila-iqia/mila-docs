003 - Multi-Node (DDP) Job
=====================================


Prerequisites:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`
* :ref:`002 - Multi-GPU Job`

Other interesting resources:
* `https://sebarnold.net/dist_blog/`
* `https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide`


Click here to see `the source code for this example <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/003_multi_node>`_

**Job.sh**

.. literalinclude:: _build/example_diffs/distributed/003_multi_node/job.sh.diff
    :language: diff

**main.py**

.. literalinclude:: _build/example_diffs/distributed/003_multi_node/main.py.diff
    :language: diff


.. .. literalinclude:: /examples/distributed/003_multi_node/job.sh
..     :language: bash

.. .. literalinclude:: /examples/distributed/003_multi_node/main.py
..     :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
