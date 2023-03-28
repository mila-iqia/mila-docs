002 - Multi-GPU Job
====================


Prerequisites:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_

Click here to see `the code for this example <https://github.com/mila-iqia/mila-docs/tree/pytorch_distributed_training_examples/docs/examples/distributed/002_multi_gpu>`_

**Job.sh**

.. literalinclude:: _build/example_diffs/distributed/002_multi_gpu/job.sh.diff
    :language: diff

.. literalinclude:: _build/example_diffs/distributed/002_multi_gpu/main.py.diff
    :language: diff


.. .. literalinclude:: /examples/distributed/002_multi_gpu/job.sh
..     :language: bash

.. .. literalinclude:: /examples/distributed/002_multi_gpu/main.py
..     :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
