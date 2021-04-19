Advanced Usage
==============

Handling preemption
-------------------

.. _advanced_preemption:

There are 2 types of preemption:

- **On the local cluster:** jobs can preempt one-another depending on their priority (unkillable>high>low) (See the `Slurm documentation <https://slurm.schedmd.com/preempt.html>`_)
- **On the cloud clusters:** virtual machines can be preempted as a limitation of less expensive virtual machines (spot/low priority)

On the local cluster, the default preemption mechanism is to killed and re-queue the job automatically without any notice. To allow a different preemption mechanism,
every partition have been duplicated (i.e. have the same characteristics as their counterparts) allowing a **120sec** grace period
before killing your job *but don't requeue it automatically*: those partitions are referred by the suffix: ``-grace`` (``main-grace, low-grace, cpu_jobs-grace``).

When using a partition with a grace period, a series of signals consisting of first ``SIGCONT`` and ``SIGTERM`` then ``SIGKILL`` will be sent to the SLURM job.
It's good practice to catch those signals using the Linux ``trap`` command to properly terminate a job and save what's necessary to restart the job.
On each cluster, you'll be allowed a *grace period* before SLURM actually kills your job (``SIGKILL``).

The easiest way to handle preemption is by trapping the ``SIGTERM`` signal

.. code-block:: bash
    :linenos:

    #SBATCH --ntasks=1
    #SBATCH ....

    exit_script() {
        echo "Preemption signal, saving myself"
        trap - SIGTERM # clear the trap
        # Optional: sends SIGTERM to child/sub processes
        kill -- -$$
    }

    trap exit_script SIGTERM

    # The main script part
    python3 my_script


.. note::
    | **Requeuing**:
    | The local Slurm cluster does not allow a grace period before preempting a job while requeuing it automatically, therefore your job will be cancelled at the end of the grace period.
    | To automatically requeue it, you can just add the ``sbatch`` command inside your ``exit_script`` function.


The following table summarizes the different preemption mode and grace periods:

====================    ==================   ==============  ===========
Cluster                      Signal(s)        Grace Period     Requeued
====================    ==================   ==============  ===========
local                    SIGCONT/SIGTERM         120s            No
Google Gloud (GCP)       SIGCONT/SIGTERM          30s            Yes
Amazon (AWS)             SIGCONT/SIGTERM         120s            Yes
Azure                          -                   -             -
====================    ==================   ==============  ===========

Packing jobs
------------

Sharing a GPU between processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``srun``, when used in a batch job is responsible for starting tasks on the allocated resources (see srun)
SLURM batch script

.. code-block:: bash
    :linenos:

    #SBATCH --ntasks-per-node=2
    #SBATCH --output=myjob_output_wrapper.out
    #SBATCH --ntasks=2
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=18G
    srun -l --output=myjob_output_%t.out python script args

this will run python 2 times, each process with 4 CPUs with the same arguments
``--output=myjob_output_%t.out`` will create 2 output files appending the task id (``%t``) to the filename and 1 global log file for things happening outside the ``srun`` command.

Knowing that, if you want to have 2 different arguments to the python program, you can use a multi-prog configuration file:
``srun -l --multi-prog silly.conf``

.. code-block::

   0  python script firstarg
   1  python script secondarg

or by specifying a range of tasks

.. code-block::

   0-1  python script %t

%t being the taskid that your python script will parse.
Note the ``-l`` on the ``srun`` command: this will prepend each line with the taskid (0:, 1:)

Sharing a node with multiple GPU 1process/GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On Compute Canada, several nodes, especially nodes with ``largeGPU`` (P100) are reserved for jobs requesting the whole node, therefore packing multiple processes in a single job can leverage faster GPU.

If you want different tasks to access different GPUs in a single allocation you need to create an allocation requesting a whole node and using ``srun`` with a subset of those resources (1 GPU).

Keep in mind that every resource not specified on the ``srun`` command while inherit the global allocation specification so you need to split each resource in a subset (except --cpu-per-task which is a per-task requirement)

Each ``srun`` represents a job step (``%s``).

Example for a GPU node with 24 cores and 4 GPUs and 128G of RAM
Requesting 1 task per GPU

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --nodes=1-1
    #SBATCH --ntasks-per-node=4
    #SBATCH --output=myjob_output_wrapper.out
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=6
    srun --gres=gpu:1 -n1 --mem=30G -l --output=%j-step-%s.out --exclusive --multi-prog python script args1 &
    srun --gres=gpu:1 -n1 --mem=30G -l --output=%j-step-%s.out --exclusive --multi-prog python script args2 &
    srun --gres=gpu:1 -n1 --mem=30G -l --output=%j-step-%s.out --exclusive --multi-prog python script args3 &
    srun --gres=gpu:1 -n1 --mem=30G -l --output=%j-step-%s.out --exclusive --multi-prog python script args4 &
    wait

This will create 4 output files:

- JOBID-step-0.out
- JOBID-step-1.out
- JOBID-step-2.out
- JOBID-step-3.out


Sharing a node with multiple GPU & multiple processes/GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combining both previous sections, we can create a script requesting a whole node with 4 GPUs, allocating 1 GPU per ``srun`` and sharing each GPU with multiple processes

Example still with a 24 cores/4 GPUs/128G RAM
Requesting 2 tasks per GPU

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --nodes=1-1
    #SBATCH --ntasks-per-node=8
    #SBATCH --output=myjob_output_wrapper.out
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=3
    srun --gres=gpu:1 -n2 --mem=30G -l --output=%j-step-%s-task-%t.out --exclusive --multi-prog silly.conf &
    srun --gres=gpu:1 -n2 --mem=30G -l --output=%j-step-%s-task-%t.out --exclusive --multi-prog silly.conf &
    srun --gres=gpu:1 -n2 --mem=30G -l --output=%j-step-%s-task-%t.out --exclusive --multi-prog silly.conf &
    srun --gres=gpu:1 -n2 --mem=30G -l --output=%j-step-%s-task-%t.out --exclusive --multi-prog silly.conf &
    wait

``--exclusive`` is important to specify subsequent step/srun to bind to different cpus.

This will produce 8 output files, 2 for each step:

- JOBID-step-0-task-0.out
- JOBID-step-0-task-1.out
- JOBID-step-1-task-0.out
- JOBID-step-1-task-1.out
- JOBID-step-2-task-0.out
- JOBID-step-2-task-1.out
- JOBID-step-3-task-0.out
- JOBID-step-3-task-1.out

Running ``nvidia-smi`` in silly.conf, while parsing the output, we can see 4 GPUs allocated and 2 tasks per GPU

.. prompt:: bash $ auto

    $ cat JOBID-step-* | grep Tesla
    0: |   0  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
    1: |   0  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
    0: |   0  Tesla P100-PCIE...  On   | 00000000:83:00.0 Off |                    0 |
    1: |   0  Tesla P100-PCIE...  On   | 00000000:83:00.0 Off |                    0 |
    0: |   0  Tesla P100-PCIE...  On   | 00000000:82:00.0 Off |                    0 |
    1: |   0  Tesla P100-PCIE...  On   | 00000000:82:00.0 Off |                    0 |
    0: |   0  Tesla P100-PCIE...  On   | 00000000:03:00.0 Off |                    0 |
    1: |   0  Tesla P100-PCIE...  On   | 00000000:03:00.0 Off |                    0 |
