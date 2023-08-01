WANDB Setup
=====================================


Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/`
* :doc:`/examples/distributed/single_gpu/`

Make sure to create a WANDB account, then you can either :
- Set your WANDB_API_KEY environment variable
- Run `wandb login` from the command line

* `<https://wandb.ai/site>`_

Other interesting resources:

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

Note : On DRAC clusters you will need to run `wandb off` to log your data as offline mode. 
You will then be able to upload your runs with the command `wandb sync --sync-all`
.. code-block:: bash

    $ wandb login

.. code-block:: bash

    $ sbatch job.sh
