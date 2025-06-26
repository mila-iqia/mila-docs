HuggingFace Accelerate example
==============================

Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`
* :doc:`/examples/distributed/multi_gpu/index`
* :doc:`/examples/distributed/multi_node/index`

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_
* `<https://huggingface.co/docs/trl/main/en/grpo_trainer#grpo-at-scale-train-a-70b-model-on-multiple-nodes>`_


Click here to see `the code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/accelerate_example>`_

**job.sh**

.. literalinclude:: job.sh
    :language: bash

**main.py**

.. literalinclude:: main.py
    :language: python

**pyproject.toml**

.. literalinclude:: pyproject.toml
    :language: toml

**Running this example**

1. Install UV from https://docs.astral.sh/uv

2. On SLURM clusters where you do not have internet access on compute nodes, you need to first create the virtual environment:

.. code-block:: bash

    $ salloc --gpus=1 --cpus-per-task=4 --mem=16G  # Get an interactive job
    $ module load httproxy/1.0  # if on a compute node, use this to get some internet access
    $ uv sync


3. Launch the job:

.. code-block:: bash

    $ sbatch job.sh
