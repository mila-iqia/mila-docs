.. _cc_clusters:

Compute Canada Clusters
#######################

.. highlight:: bash


Account Creation
----------------

To access the Compute Canada (CC) clusters you have to first create an account at https://ccdb.computecanada.ca. Use a password with at least 8 characters, mixed case letters, digits and special characters. Later you will be asked to create another password with those rules, and it’s really convenient that the two password are the same.

Then, you have to apply for a ``role`` at https://ccdb.computecanada.ca/me/add_role, which basically means telling CC that you are part of the lab so they know which cluster you can have access to, and track your usage.

You will be asked for the CCRI (See screenshot below). Please reach out to your sponsor to get the CCRI.

.. image:: role.png
    :align: center
    :alt: role.png

You will need to **wait** for your sponsor to accept before being able to login to the CC clusters.

Clusters
--------

.. _beluga:

Beluga
^^^^^^

Beluga is a cluster located at ETS in Montreal. It uses `Slurm` to schedule jobs. Its full documentation can be found `here <https://docs.computecanada.ca/wiki/Béluga/en>`__, and its current status `here <http://status.computecanada.ca>`__.

You can access Beluga via ssh:

.. prompt:: bash $

    ssh <user>@beluga.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).


Launching Jobs
""""""""""""""

Users must specify the resource allocation Group Name using the flag ``--account=rrg-bengioy-ad``.
To launch a CPU-only job:

.. prompt:: bash $

    sbatch --time=1:0:0 --account=rrg-bengioy-ad job.sh

To launch a GPU job:

.. prompt:: bash $

    sbatch --time=1:0:0 --account=rrg-bengioy-ad --gres=gpu:1 job.sh

And to get an interactive session, use the ``salloc`` command:

.. prompt:: bash $

    salloc --time=1:0:0 --account=rrg-bengioy-ad --gres=gpu:1

The full documentation for jobs launching on Beluga can be found `here <https://docs.computecanada.ca/wiki/Running_jobs>`__.


Nodes
"""""

The GPU nodes consist of:

* 40 CPU cores
* 186 GB RAM
* 4 GPU NVIDIA V100 (16GB)

.. tip:: You should ask for max 10 CPU cores and 32 GB of RAM per GPU you are requesting (as explained `here <https://docs.computecanada.ca/wiki/Allocations_and_resource_scheduling>`__), otherwise, your job will count for more than 1 allocation, and will take more time to get scheduled.


.. _cc_storage:

Storage
"""""""

============== ==================== =========================
Storage        Path                 Usage
============== ==================== =========================
$HOME          /home/<user>/        * Code
                                    * Specific libraries
$HOME/projects /project/rpp-bengioy * Compressed raw datasets
$SCRATCH       /scratch/<user>      * Processed datasets
                                    * Experimental results
                                    * Logs of experiments
$SLURM_TMPDIR                       * Temporary job results
============== ==================== =========================

They are roughly listed in order of increasing performance and optimized for different uses:

* The ``$HOME`` folder on NFS is appropriate for codes and libraries which are small and read once. **Do not write experiemental results here!**
* The ``/projects`` folder should only contain **compressed raw** datasets (**processed** datasets should go in ``$SCRATCH``). We have a limit on the size and number of file in ``/projects``, so do not put anything else there. If you add a new dataset there (make sure it is readable by every member of the group using ``chgrp -R rpp-bengioy <dataset>``).
* The ``$SCRATCH`` space can be used for short term storage. It has good performance and large quotas, but is purged regularly (every file that has not been used in the last 3 months gets deleted, but you receive an email before this happens).
* ``$SLURM_TMPDIR`` points to the local disk of the node on which a job is running. It should be used to copy the data on the node at the beginning of the job and write intermediate checkpoints. This folder is cleared after each job.

When an experiment is finished, results should be transferred back to Mila servers.

More details on storage can be found `here <https://docs.computecanada.ca/wiki/B%C3%A9luga/en#Storage>`__.


Modules
"""""""

Many software, such as Python or MATLAB are already compiled and available on Beluga through the ``module`` command and its subcommands. Its full documentation can be found `here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en>`__.

====================== =====================================
module avail           Displays all the available modules
module load <module>   Loads <module>
module spider <module> Shows specific details about <module>
====================== =====================================

In particular, if you with to use ``Python 3.6`` you can simply do:

.. prompt:: bash $

    module load python/3.6

.. tip:: If you wish to use Python on the cluster, we strongly encourage you to read `CC Python Documentation <https://docs.computecanada.ca/wiki/Python>`_, and in particular the `Pytorch <https://docs.computecanada.ca/wiki/PyTorch>`_ and/or `Tensorflow <https://docs.computecanada.ca/wiki/TensorFlow>`_ pages.

The cluster has many python packages (or ``wheels``), such already compiled for the cluster. See `here <https://docs.computecanada.ca/wiki/Python/en>`__ for the details. In particular, you can browse the packages by doing:

.. prompt:: bash $

    avail_wheels <wheel>

Such wheels can be installed using pip. Moreover, the most efficient way to use modules on the cluster is to `build your environnement inside your job <https://docs.computecanada.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs>`_. See the script example below.


Script Example
""""""""""""""

Here is a ``sbatch`` script that follows good practices on Beluga:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
    #SBATCH --cpus-per-task=6                # Ask for 6 CPUs
    #SBATCH --gres=gpu:1                     # Ask for 1 GPU
    #SBATCH --mem=32G                        # Ask for 32 GB of RAM
    #SBATCH --time=3:00:00                   # The job will run for 3 hours
    #SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH

    # 1. Create your environement locally
    module load python/3.6
    virtualenv --no-download $SLURM_TMPDIR/env
    source $SLURM_TMPDIR/env/bin/activate
    pip install --no-index torch torchvision

    # 2. Copy your dataset on the compute node
    # IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
    cp $SCRATCH/<dataset.zip> $SLURM_TMPDIR

    # 3. Eventually unzip your dataset
    unzip $SLURM_TMPDIR/<dataset.zip> -d $SLURM_TMPDIR

    # 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
    #    and look for the dataset into $SLURM_TMPDIR
    python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

    # 5. Copy whatever you want to save on $SCRATCH
    cp $SLURM_TMPDIR/<to_save> $SCRATCH

Using CometML and Wandb
"""""""""""""""""""""""

The compute nodes for Beluga don't have access to the internet,
but there is a special module that can be loaded in order to allow
training scripts to access some specific servers, which includes
the necessary servers for using CometML and Wandb ("Weights and Biases").

.. prompt:: bash $

    module load httpproxy

More documentation about this can be found at `https://docs.computecanada.ca/wiki/Weights_%26_Biases_(wandb)`.


.. _graham:

Graham
^^^^^^

Graham is a cluster located at University of Waterloo. It uses SLURM to schedule jobs. Its full documentation can be found `here <https://docs.computecanada.ca/wiki/Graham/>`__, and its current status `here <http://status.computecanada.ca>`__.

You can access Graham via ssh:

.. prompt:: bash $

    ssh <user>@graham.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).

Since its structure is similar to `Beluga`, please look at the `Beluga`_ documentation, as well as relevant parts of the `Compute Canada Documentation <https://docs.computecanada.ca/wiki/Graham>`__.

.. note:: For GPU jobs the ressource allocation Group Name is the same as Beluga, so you should use the flag ``--account=rrg-bengioy-ad`` for GPU jobs.


.. _cedar:

Cedar
^^^^^

Cedar is a cluster located at Simon Fraser University. It uses SLURM to schedule jobs. Its full documentation can be found `here <https://docs.computecanada.ca/wiki/Cedar>`__, and its current status `here <http://status.computecanada.ca>`__.

You can access Cedar via ssh:

.. prompt:: bash $

    ssh <user>@cedar.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).

Since its structure is similar to `Beluga`, please look at the `Beluga`_ documentation, as well as relevant parts of the `Compute Canada Documentation <https://docs.computecanada.ca/wiki/Cedar>`__.

.. note:: However, we don't have any CPU priority on Cedar, in this case you can use ``--account=def-bengioy`` for CPU. Thus, it might take some time before they start.



FAQ
---

What to do with  `ImportError: /lib64/libm.so.6: version GLIBC_2.23 not found`?
    The structure of the file system is different than a classical Linux, so your code has trouble finding libraries. See `how to install binary packages <https://docs.computecanada.ca/wiki/Installing_software_in_your_home_directory#Installing_binary_packages>`_.

Disk quota exceeded error on ``/project`` file systems
    You have files in ``/project`` with the wrong permissions. See `how to change permissions <https://docs.computecanada.ca/wiki/Frequently_Asked_Questions/en#Disk_quota_exceeded_error_on_.2Fproject_filesystems>`_.

