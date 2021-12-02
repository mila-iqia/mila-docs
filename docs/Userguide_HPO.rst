Hyperparameter Optimization
===========================

Hyperparameter optimization is very easy to parallelize as each trial (unique set of hyperparameters) are
independant of each other. The easiest way is to launch as many jobs as possible each trying a different
set of hyperparameters and reporting their results back to a synchronized location (database).

You will need to estimate how much resources your training requires and update the provided example to fit.
In the example below we use 100 tasks with each 4 CPU cores and one GPU.
Each task will run 4 training in parallel on the same GPU to maximize its utilization.

This means there could be 400 set of hyperparameters being worked on in parallel across 100 GPUs.


.. image:: _static/hpo_diagram.png


.. code-block:: bash

   # launch 100 jobs (from 0-99)
   # that will work in parallel to find the optimal hyperparameters
   #SBATCH --array=0-100
   #SBATCH --ntasks=1

   # Each node will have 1 GPU with 4 CPU-cores
   #SBATCH --gres=gpu:1
   #SBATCH --mem=16Go
   #SBATCH --cpus-per-gpu=4

   export SCRATCH=/network/scratch/
   export EXPERIMENT_NAME='MySuperExperiment'
   export SEARCH_SPACE=$SLURM_TMPDIR/search-space.json 
   export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml

   # Configure Orion
   #    - user hyperband
   #    - launch 4 workers for each tasks (one for each CPU)
   #    - worker dies if idle for more than a minute
   #
   cat > $ORION_CONFIG <<- EOM
       experiment:
           algorithms:
               hyperband:
                   seed: None
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
   EON

   # Configure the experiment search space
   cat > $SEARCH_SPACE <<- EOM
       {
           "lr": "orion~loguniform(1e-5, 1.0)",
       }
   EOM

   orion --config $ORION_CONFIG hunt --config $SEARCH_SPACE python ./train.py --cuda


.. note::

    Do not forget to move orion database out of ``scratch`` once the experiment is done.
