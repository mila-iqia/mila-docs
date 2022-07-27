.. _milacluster_storage:


Storage
=======


====================================================== =========== ====================================== =================== ============
Path                                                   Performance Usage                                  Quota (Space/Files) Auto-cleanup
====================================================== =========== ====================================== =================== ============
``$HOME`` or ``/home/mila/<u>/<username>/``            Low         * Personal user space                  100G/1000K
                                                                   * Specific libraries, code, binaries
``$SCRATCH`` or ``/network/scratch/<u>/<username>/``   High        * Temporary job results                                    90 days
                                                                   * Processed datasets
                                                                   * Optimized for small Files
``$SLURM_TMPDIR``                                      Highest     * High speed disk for temporary job    4T/-                at job end
                                                                     results
``/network/projects/<groupname>/``                     Fair        * Shared space to facilitate           200G/1000K
                                                                     collaboration between researchers
                                                                   * Long-term project storage
``/network/datasets/``                                 High        * Curated raw datasets (read only)
====================================================== =========== ====================================== =================== ============

.. note:: The ``$HOME`` file system is backed up once a day. For any file
   restoration request, file a request to `Mila's IT support
   <https://it-support.mila.quebec>`_ with the path to the file or directory to
   restore, with the required date.

.. warning:: Currently there is no backup system for any other file systems of
   the Mila cluster. Storage local to personal computers, Google Drive and other
   related solutions should be used to backup important data

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

datasets
--------

``datasets`` contains curated datasets to the benefit of the Mila community.
To request the addition of a dataset or a preprocessed dataset you think could
benefit the research of others, you can fill `this form
<https://forms.gle/vDVwD2rZBmYHENgZA>`_.

Datasets in ``datasets/restricted`` are restricted and require an explicit
request to gain access. Please `submit a support ticket
<https://milaquebec.freshdesk.com/a/tickets/new>`_ mentioning the dataset's
access group (ex.: ``scannet_users``), your cluster's username and the
approbation of the group owner. You can find the dataset's access group by
listing the content of ``/network/datasets/restricted`` with the `ls command
<https://cli-cheatsheet.readthedocs.io/en/latest/#ls>`_.
