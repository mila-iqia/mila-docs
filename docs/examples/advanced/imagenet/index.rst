Multi-Node / Multi-GPU ImageNet Training
========================================


Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`
* :doc:`/examples/distributed/multi_gpu/index`
* :doc:`/examples/distributed/multi_node/index`
* :doc:`/examples/good_practices/checkpointing/index`
* :doc:`/examples/good_practices/launch_many_jobs/index`

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_


Click here to see `the source code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/advanced/imagenet>`_

**job.sh**

.. literalinclude:: job.sh
    :language: bash

**pyproject.toml**

.. literalinclude:: pyproject.toml
    :language: toml

**main.py**

.. literalinclude:: main.py
    :language: python


**prepare_data.py**

.. literalinclude:: prepare_data.py
    :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
