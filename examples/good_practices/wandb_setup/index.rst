Wandb Setup
=====================================


Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`

Make sure to create a Wandb account, then you can either :

* Set your ``WANDB_API_KEY`` environment variable
* Run ``wandb login`` from the command line

Other resources:

* `<https://docs.wandb.ai/quickstart>`_

Click here to see `the source code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/wandb_setup>`_

**job.sh**

.. literalinclude:: job.sh.diff
    :language: diff


**main.py**

.. literalinclude:: main.py.diff
    :language: diff


**Running this example**

Note : On DRAC clusters you will need to run ``wandb off`` to log your data as offline mode.
You will then be able to upload your runs with the command ``wandb sync --sync-all``

.. code-block:: bash

    $ wandb login

.. code-block:: bash

    $ sbatch job.sh
