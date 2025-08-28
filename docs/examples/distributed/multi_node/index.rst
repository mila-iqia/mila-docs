Multi-Node (DDP) Job
====================


Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`
* :doc:`/examples/distributed/multi_gpu/index`

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_


Click here to see `the source code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/multi_node>`_

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

.. code-block:: bash

    $ sbatch job.sh
