.. _milacluster_storage:


Storage
=======


====================================================== =========== ====================================== =================== ====== ============
Path                                                   Performance Usage                                  Quota (Space/Files) Backup Auto-cleanup
====================================================== =========== ====================================== =================== ====== ============
``/network/datasets/``                                 High        * Curated raw datasets (read only)
``$HOME`` or ``/home/mila/<u>/<username>/``            Low         * Personal user space                  100GB/1000K         Daily  no
                                                                   * Specific libraries, code, binaries
``$SCRATCH`` or ``/network/scratch/<u>/<username>/``   High        * Temporary job results                no                  no     90 days
                                                                   * Processed datasets
                                                                   * Optimized for small Files
``$SLURM_TMPDIR``                                      Highest     * High speed disk for temporary job    4TB/-               no     at job end
                                                                     results
``/network/projects/<groupname>/``                     Fair        * Shared space to facilitate           200GB/1000K         Daily  no
                                                                     collaboration between researchers
                                                                   * Long-term project storage
``$ARCHIVE`` or ``/network/archive/<u>/<username>/``   Low         * Long-term personal storage           500GB               no     no
====================================================== =========== ====================================== =================== ====== ============

.. warning:: Currently there is no backup system for the Mila cluster. Storage
   local to personal computers, Google Drive and other related solutions should
   be used to backup important data

$HOME
-----

``$HOME`` is appropriate for codes and libraries which are small and read once,
as well as the experimental results that would be needed at a later time (e.g.
the weights of a network referenced in a paper).

Quotas are enabled on ``$HOME`` for both disk capacity (blocks) and number of
files (inodes). The limits for blocks and inodes are respectively 100GiB and 1
million per user. The command to check the quota usage from a login node is:

.. prompt:: bash $

   beegfs-ctl --cfgFile=/etc/beegfs/home.d/beegfs-client.conf --getquota --uid $USER

$SCRATCH
--------

``$SCRATCH`` can be used to store processed datasets, work in progress datasets
or temporary job results. Its block size is optimized for small files which
minimizes the performance hit of working on extracted datasets.

.. note:: **Auto-cleanup**: this file system is cleared on a weekly basis,
   files not used for more than 90 days will be deleted.

$SLURM_TMPDIR
-------------

``$SLURM_TMPDIR`` points to the local disk of the node on which a job is
running. It should be used to copy the data on the node at the beginning of the
job and write intermediate checkpoints. This folder is cleared after each job.

projects
--------

``projects`` can be used for collaborative projects. It aims to ease the
sharing of data between users working on a long-term project.

Quotas are enabled on ``projects`` for both disk capacity (blocks) and number
of files (inodes). The limits for blocks and inodes are respectively 200GiB and
1 million per user and per group.

.. note:: It is possible to request higher quota limits if the project requires
   it. File a request to `Mila's IT support <https://it-support.mila.quebec>`_.

$ARCHIVE
--------

``$ARCHIVE`` purpose is to store data other than datasets that has to be kept
long-term (e.g.  generated samples, logs, data relevant for paper submission).

``$ARCHIVE`` is only available on the **login** nodes. Because this file system
is tuned for large files, it is recommended to archive your directories. For
example, to archive the results of an experiment in
``$SCRATCH/my_experiment_results/``, run the commands below from a login node:

.. prompt:: bash $

   cd $SCRATCH
   tar cJf $ARCHIVE/my_experiment_results.tar.xz --xattrs my_experiment_results

Disk capacity quotas are enabled on ``$ARCHIVE``. The soft limit per user is
500GB, the hard limit is 550GB. The grace time is 7 days. This means that one
can use more than 500GB for 7 days before the file system enforces quota.
However, it is not possible to use more than 550GB.
The command to check the quota usage from a login node is `df`:

.. prompt:: bash $

   df -h $ARCHIVE

.. note:: There is **NO** backup of this file system.

datasets
--------

``datasets`` contains curated datasets to the benefit of the Mila community.
To request the addition of a dataset or a preprocessed dataset you think could
benefit the research of others, you can fill `this form
<https://forms.gle/vDVwD2rZBmYHENgZA>`_.
