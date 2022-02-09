Hyperparameter Optimization
===========================

Hyperparameter optimization is very easy to parallelize as each trial (unique set of hyperparameters) are
independant of each other. The easiest way is to launch as many jobs as possible each trying a different
set of hyperparameters and reporting their results back to a synchronized location (database).

You will need to estimate how much resources your training requires and update the provided example to fit.
In the example below we use 100 tasks with each 4 CPU cores and one GPU.
Each task will run 4 training in parallel on the same GPU to maximize its utilization.

This means there could be 400 set of hyperparameters being worked on in parallel across 100 GPUs.

The easiest way to run an hyperparameter search on the cluster is simply to use a job array,
which will launch the same job n times. Your HPO library will generate different parameters to try
for each instances.

.. code-block

    sbatch --array=1-100 --gres=gpu:1 --cpus-per-gpu=2 --mem-per-gpu=16G scripts/hpo_launcher.sh train.py

Configure Orion
^^^^^^^^^^^^^^^

`Orion <https://orion.readthedocs.io/en/stable/?badge=stable>`_
is an asynchronous framework for black-box function optimization developped at Mila.

Its purpose is to serve as a meta-optimizer for machine learning models and training,
as well as a flexible experimentation platform for large scale asynchronous optimization procedures.

Orion saves all the results of its optimization process in a database,
by default it is using a local database on a shared filesystem named ``pickleddb``.
You will need to specify its location and the name of your experiment.
Optionally you can configure workers which will run on parallel to maximize resource usage.

.. code-block:: bash

   cat > $ORION_CONFIG <<- EOM
       experiment:
           name: ${EXPERIMENT_NAME}
           algorithms:
               hyperband:
                   seed: null
           max_broken: 10

       worker:
           n_workers: $SBATCH_CPUS_PER_GPU
           pool_size: 0
           executor: joblib
           heartbeat: 120
           max_broken: 10
           idle_timeout: 60

       database:
           host: $SCRATCH/${EXPERIMENT_NAME}_orion.pkl
           type: pickleddb
   EOM

Define the search space
^^^^^^^^^^^^^^^^^^^^^^^

We now need to define a search space that Orion will go through.
The search space is going to be dependent on your model, optimizer and others.
You find below an example of a common set of hyperparameters.

.. code-block:: bash

   # Define your hyper parameter search space
   cat > $SPACE_CONFIG <<- EOM
      {
         "epochs": "orion~fidelity(1, 100, base=2)",
         "lr": "orion~loguniform(1e-5, 1.0)",
         "weight_decay": "orion~loguniform(1e-10, 1e-3)",
         "momentum": "orion~loguniform(0.9, 1.0)"
      }
   EOM

Run Orion hunt
^^^^^^^^^^^^^^

Now that orion is configured and has a search space we can execute orion-hunt
which will start training with a different set of hyper parameters for each worker.


.. code-block:: bash

   orion hunt --config $ORION_CONFIG python train.py --config $SPACE_CONFIG --arg1 value1


.. note::

   Do not forget to move orion database out of ``scratch`` once the experiment is done.


Full Example
^^^^^^^^^^^^

.. code-block:: bash

   sbatch --array=1-100 --gres=gpu:1 --cpus-per-gpu=2 --mem-per-gpu=16G hpo_launcher.sh train.py


.. literalinclude:: /examples/hpo_launcher.sh
   :language: bash
   :linenos:

.. note::

   Network Architecture Search (NAS) is a subset of hyperparameter optimization where hyper-parameters
   are used to generate a network. You can use the same technique for both NAS and HPO.
