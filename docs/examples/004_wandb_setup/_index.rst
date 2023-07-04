004 - WANDB Setup
=====================================


Prerequisites:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

Make sure to create a WANDB account and set your WANDB_API_KEY environment variable.
* `<https://wandb.ai/site>`_

Other interesting resources:

* `<https://docs.wandb.ai/quickstart>`_

Click here to see `the source code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/004_wandb_setup>`_

**Job.sh**

.. literalinclude:: examples/distributed/004_wandb_setup/job.sh.diff
    :language: diff

**main.py**

.. literalinclude:: examples/distributed/004_wandb_setup/main.py.diff
    :language: diff


.. .. literalinclude:: examples/distributed/004_wandb_setup/job.sh
..     :language: bash

.. .. literalinclude:: examples/distributed/004_wandb_setup/main.py
..     :language: python


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
