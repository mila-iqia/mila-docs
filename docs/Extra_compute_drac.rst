.. highlight:: bash
.. _drac_clusters:


Digital Research Alliance of Canada Clusters
============================================

The clusters named `Beluga`, `Cedar`, `Graham`, `Narval` and `Niagara` are
clusters provided by the `Digital Research Alliance of Canada organisation
<https://alliancecan.ca/>`_ (the Alliance). For Mila researchers, these
clusters are to be used for larger experiments having many jobs, multi-node
computation and/or multi-GPU jobs as well as long running jobs.

.. note::

   If you use DRAC resources for your research, please remember to `acknowledge
   their use in your papers
   <https://alliancecan.ca/en/services/advanced-research-computing/acknowledging-alliance>`_

.. note::

   Compute Canada ceased its operational responsibilities for supporting Canada’s
   national advanced research computing (ARC) platform on March 31, 2022. The services
   will be supported by the new Digital Research Alliance of Canada.

   https://ace-net.ca/compute-canada-operations-move-to-the-digital-research-alliance-of-canada-(the-alliance).html

Current allocation description
------------------------------

Clusters of the Alliance are shared with researchers across the country.
Allocations are given by the Alliance to selected research groups to ensure to
a minimal amount of computational resources throughout the year.

Depending on your affiliation, you will have access to different allocations. If
you are a student at University of Montreal, you can have access to the
``rrg-bengioy-ad`` allocation described below. For students from other
universities, you should ask your advisor to know which allocations you could
have access to.

From the Alliance's documentation: `An allocation is an amount of resources
that a research group can target for use for a period of time, usually a year.`
To be clear, it is not a maximal amount of resources that can be used
simultaneously, it is a weighting factor of the workload manager to balance
jobs. For instance, even though we are allocated 408 GPU-years across all
clusters, we can use more or less than 408 GPUs simultaneously depending on the
history of usage from our group and other groups using the cluster at a given
period of time. Please see the Alliance's `documentation
<https://docs.alliancecan.ca/wiki/Allocations_and_resource_scheduling>`__ for
more information on how allocations and resource scheduling are configured for
these installations.

.. Il est possiblement dangeureux de donner le nom de compte de Yoshua sur un
   site publiquement disponible.

The table below provides information on the allocation for
``rrg-bengioy-ad`` for the period which spans from April 2022 to
April 2023. Note that there are no special allocations for GPUs on
Graham and therefore jobs with GPUs should be submitted with the
account ``def-bengioy``.


+------------------------+-----------------------+---------------------------------------------------------+
| Cluster                | CPUs                  | GPUs                                                    |
|                        +------+----------------+-----------+-----+----------------------+----------------+
|                        |  #   | account        | Model     | #   | SLURM type specifier | account        |
+------------------------+------+----------------+-----------+-----+----------------------+----------------+
| :ref:`Beluga <beluga>` |  197 | rrg-bengioy-ad | V100-16G  | 127 | ``v100``             | rrg-bengioy-ad |
+------------------------+------+----------------+-----------+-----+----------------------+----------------+
| :ref:`Cedar <cedar>`   |  197 | rrg-bengioy-ad | V100-32G  | 127 | ``v100l``            | rrg-bengioy-ad |
+------------------------+------+----------------+-----------+-----+----------------------+----------------+
| :ref:`Narval <narval>` |  917 | rrg-bengioy-ad | A100-40G  | 154 | ``a100``             | rrg-bengioy-ad |
+------------------------+------+----------------+-----------+-----+----------------------+----------------+



Account Creation
----------------

To access the Alliance clusters you have to first create an account at
https://ccdb.alliancecan.ca. Use a password with at least 8 characters, mixed
case letters, digits and special characters. Later you will be asked to create
another password with those rules, and it’s really convenient that the two
password are the same.

Then, you have to apply for a ``role`` at
https://ccdb.alliancecan.ca/me/add_role, which basically means telling the
Alliance that you are part of the lab so they know which cluster you can have
access to, and track your usage.

You will be asked for the CCRI (See screenshot below). Please reach out to your
sponsor to get the CCRI.

.. image:: role.png
    :align: center
    :alt: role.png

You will need to **wait** for your sponsor to accept before being able to login
to the Alliance clusters.

You should apply to a ``role`` using this form **for each allocation you can have access to**. If, for instance,
your supervisor is member of the ``rrg-bengioy-ad`` allocation, you should apply using Yoshua Bengio's CCRI, and
you should apply separately using your supervisor's CCRI to have access to ``def-<yoursupervisor>``. Ask your supervisor
to share these CCRI with you.

Clusters
--------

Narval:
   (:ref:`Mila doc <narval>`)
   (`Digital Research Alliance of Canada doc <https://docs.alliancecan.ca/wiki/Narval/en>`__)

   For most students, Narval is the best choice for both CPU and GPU jobs because
   of larger allocations on this cluster.
   Narval is also the newest cluster, and contains the most powerful GPUs (A100). If your
   job can benefit from the A100's features, such as TF32 floating-point math, Narval
   is the best choice.
Beluga:
   (:ref:`Mila doc <beluga>`)
   (`Digital Research Alliance of Canada doc <https://docs.alliancecan.ca/wiki/B%C3%A9luga/en>`__)

   Beluga is a good alternative for CPU and GPU jobs.
Cedar:
   (:ref:`Mila doc <cedar>`)
   (`Digital Research Alliance of Canada doc <https://docs.alliancecan.ca/wiki/Cedar/en>`__)

   Cedar is a good alternative to Beluga if you absolutely need to have an internet connection
   on the compute nodes.
Graham:
   (:ref:`Mila doc <graham>`)
   (`Digital Research Alliance of Canada doc <https://docs.alliancecan.ca/wiki/Graham/en>`__)

   We do not have any CPU or GPU allocation on Graham anymore, but you can use it with `def-<supervisor>`
   if other clusters are overcrowded. (where `<supervisor> is the DRAC account name of your supervisor`)
Niagara:
   (:ref:`Mila doc <niagara>`)
   (`Digital Research Alliance of Canada doc <https://docs.alliancecan.ca/wiki/Niagara/en>`__)

   Niagara is not recommended for most students. It is a CPU-only cluster with unusual
   configurations. Access is not automatic; It is opt-in and must be requested via
   CCDB manually. Compute resources in Niagara are not assigned to jobs on a per-CPU,
   but on a per-node basis.


Narval
^^^^^^

Narval is a cluster located at `ÉTS <https://www.etsmtl.ca/>`_ in Montreal. It
uses SLURM to schedule jobs. Its full documentation can be found
`here <https://docs.alliancecan.ca/wiki/Narval>`__, and its current status
`here <http://status.alliancecan.ca>`__.

You can access Narval via ssh:

.. prompt:: bash $

    ssh <user>@narval.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).

While Narval has a filesystem organization similar to the other clusters, and the
newest GPUs in the fleet (A100s), it differs from the other clusters in that it
uses AMD CPUs (Zen 2/3) rather than Intel (Broadwell/Skylake). This *may* (but is
not guaranteed to) result in performance or behaviour differences, up to and
including hangs.

.. warning::

    A very notable difference in the feature-set of Narval's CPUs is that the
    AMD CPUs of this cluster do **not** support the AVX-512 vector extensions,
    while the Intel CPUs of the older clusters **do**. This makes it unsafe to
    run *compiled* CPU code from older Intel-based clusters to Narval, but the
    opposite (although ill-advised) will work. The symptom of attempting to
    execute AVX-512 code on Narval's CPUs is that the program fatally aborts
    with signal ``SIGILL`` and messages such as ``Illegal instruction``.


Launching Jobs
""""""""""""""

Users must specify the resource allocation Group Name using the flag
``--account=rrg-bengioy-ad``.  To launch a CPU-only job:

.. prompt:: bash $

   sbatch --time=1:00:00 --account=rrg-bengioy-ad job.sh

.. note::

   The account name will differ based on your affiliation.

To launch a GPU job:

.. prompt:: bash $

    sbatch --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1 job.sh

And to get an interactive session, use the ``salloc`` command:

.. prompt:: bash $

    salloc --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1

The full documentation for jobs launching on Alliance clusters can be found
`here <https://docs.alliancecan.ca/wiki/Running_jobs>`__.


Narval nodes description
""""""""""""""""""""""""

Each GPU node consists of:

* 48 CPU cores
* 498 GB RAM
* 4 GPU NVIDIA A100 (40GB)

.. tip:: You should ask for max 12 CPU cores and 124 GB of RAM per GPU you are
   requesting (as explained `here
   <https://docs.alliancecan.ca/wiki/Allocations_and_resource_scheduling>`__),
   otherwise, your job will count for more than 1 allocation, and will take
   more time to get scheduled.


.. _drac_storage:


Narval Storage
""""""""""""""

================== ======================= =========================
Storage            Path                    Usage
================== ======================= =========================
``$HOME``          /home/<user>/           * Code
                                           * Specific libraries
``$HOME/projects`` /project/rrg-bengioy-ad * Compressed raw datasets
``$SCRATCH``       /scratch/<user>         * Processed datasets
                                           * Experimental results
                                           * Logs of experiments
``$SLURM_TMPDIR``                          * Temporary job results
================== ======================= =========================

They are roughly listed in order of increasing performance and optimized for
different uses:

* The ``$HOME`` folder on Lustre is appropriate for code and libraries, which
  are small and read once. **Do not write experiemental results here!**
* The ``$HOME/projects`` folder should only contain **compressed raw** datasets
  (**processed** datasets should go in ``$SCRATCH``). We have a limit on the
  size and number of file in ``$HOME/projects``, so do not put anything else
  there.  If you add a new dataset there (make sure it is readable by every
  member of the group using ``chgrp -R rpp-bengioy <dataset>``).
* The ``$SCRATCH`` space can be used for short term storage. It has good
  performance and large quotas, but is purged regularly (every file that has
  not been used in the last 3 months gets deleted, but you receive an email
  before this happens).
* ``$SLURM_TMPDIR`` points to the local disk of the node on which a job is
  running. It should be used to copy the data on the node at the beginning of
  the job and write intermediate checkpoints. This folder is cleared after each
  job, so results there must be copied to ``$SCRATCH`` at the end of a job.

When a series of experiments is finished, results should be transferred back
to Mila servers.

More details on storage can be found `here
<https://docs.alliancecan.ca/wiki/Narval/en#Storage>`__.


Modules
"""""""

Many software, such as Python or MATLAB are already compiled and available on
Beluga through the ``module`` command and its subcommands. Its full
documentation can be found `here
<https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en>`__.

====================== =====================================
module avail           Displays all the available modules
module load <module>   Loads <module>
module spider <module> Shows specific details about <module>
====================== =====================================

In particular, if you with to use ``Python 3.6`` you can simply do:

.. prompt:: bash $

    module load python/3.6

.. tip:: If you wish to use Python on the cluster, we strongly encourage you to
   read `Alliance Python Documentation
   <https://docs.alliancecan.ca/wiki/Python>`_, and in particular the `Pytorch
   <https://docs.alliancecan.ca/wiki/PyTorch>`_ and/or `Tensorflow
   <https://docs.alliancecan.ca/wiki/TensorFlow>`_ pages.

The cluster has many Python packages (or ``wheels``), such already compiled for
the cluster. See `here <https://docs.alliancecan.ca/wiki/Python/en>`__ for the
details. In particular, you can browse the packages by doing:

.. prompt:: bash $

    avail_wheels <wheel>

Such wheels can be installed using pip. Moreover, the most efficient way to use
modules on the cluster is to `build your environnement inside your job
<https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs>`_.
See the script example below.


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

The compute nodes for Narval, Graham and Beluga don't have access to the
internet, but there is a special module that can be loaded in order to allow
training scripts to access some specific servers, which includes
the necessary servers for using CometML and Wandb ("Weights and Biases").

.. prompt:: bash $

    module load httpproxy

More documentation about this can be found `here
<https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb)>`__.

.. note::

   Be careful when using Wandb with `httpproxy`. It does not support sending
   artifacts and wandb's logger will hang in the background when your training
   is completed, wasting ressources until the job times out. It is recommended
   to use the offline mode with wandb instead to avoid such waste.


Beluga
^^^^^^

Beluga is a cluster located at the ÉTS (École de Technologie Supérieure) in
Montreal. It uses SLURM to schedule jobs. Its full documentation can be found
`here <https://docs.alliancecan.ca/wiki/B%C3%A9luga/en>`__, and its current
status `here <http://status.alliancecan.ca>`__.

You can access Beluga via ssh:

.. prompt:: bash $

   ssh <user>@beluga.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).


Beluga nodes description
""""""""""""""""""""""""

Each GPU node consists of:

* 40 CPU cores
* 186 GB RAM
* 4 GPU NVIDIA V100 (16GB)

.. tip:: You should ask for max 10 CPU cores and 32 GB of RAM per GPU you are
   requesting (as explained `here
   <https://docs.alliancecan.ca/wiki/Allocations_and_resource_scheduling>`__),
   otherwise, your job will count for more than 1 allocation, and will take
   more time to get scheduled.


Graham
^^^^^^

Graham is a cluster located at University of Waterloo. It uses SLURM to schedule
jobs. Its full documentation can be found `here
<https://docs.alliancecan.ca/wiki/Graham/>`__, and its current status `here
<http://status.alliancecan.ca>`__.

You can access Graham via ssh:

.. prompt:: bash $

    ssh <user>@graham.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).



Cedar
^^^^^

Cedar is a cluster located at Simon Fraser University. It uses SLURM to schedule
jobs. Its full documentation can be found `here
<https://docs.alliancecan.ca/wiki/Cedar>`__, and its current status `here
<http://status.alliancecan.ca>`__.

You can access Cedar via ssh:

.. prompt:: bash $

    ssh <user>@cedar.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).



Niagara
^^^^^^^

Niagara is a cluster located at the University of Toronto. It uses SLURM to
schedule jobs. Its full documentation can be found `here
<https://docs.alliancecan.ca/wiki/Niagara>`__, and its current status `here
<http://status.alliancecan.ca>`__.

You can access Niagara via ssh:

.. prompt:: bash $

    ssh <user>@niagara.computecanada.ca

Where ``<user>`` is the username you created previously (see `Account Creation`_).

Niagara is completely unlike the previous clusters, as mentioned above. Access
to it is opt-in, it has no GPUs, allocations are *only* per-**node** and *never*
per-CPU-core, and the software environment is different. You are very unlikely
to need this cluster and are strongly encouraged to peruse its documentation
if you have a strong reason to use it regardless. Do not expect to be able to
schedule and run CPU jobs on Niagara exactly the same way as on all other clusters.


FAQ
---

What to do with  `ImportError: /lib64/libm.so.6: version GLIBC_2.23 not found`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The structure of the file system is different than a classical Linux, so your
code has trouble finding libraries. See `how to install binary packages
<https://docs.alliancecan.ca/wiki/Installing_software_in_your_home_directory#Installing_binary_packages>`_.

Disk quota exceeded error on ``/project`` file systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have files in ``/project`` with the wrong permissions. See `how to change
permissions
<https://docs.alliancecan.ca/wiki/Frequently_Asked_Questions/en#Disk_quota_exceeded_error_on_.2Fproject_filesystems>`_.

