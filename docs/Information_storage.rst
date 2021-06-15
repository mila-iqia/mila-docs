.. _milacluster_storage:

Storage
=======

.. TODO:: Cette table doit être mise à jour.


=============================== ========================================= =================== ============
Path                            Usage                                     Quota (Space/Files) Auto-cleanup
=============================== ========================================= =================== ============
/home/mila/<u>/<username>/         * Personal user space                       200G/1000K
                                   * Specific libraries/code
/miniscratch/                      * Temporary job results                                      3 months
/network/projects/<groupname>/     * Long-term project storage                 200G/1000K
/network/data1/                    * Raw datasets (read only)
/network/datasets/                 * Curated raw datasets (read only)
$SLURM_TMPDIR                      * High speed disk for                                       at job end
                                     temporary job results
=============================== ========================================= =================== ============

* The ``home`` folder is appropriate for codes and libraries which are small and
  read once, as well as the experimental results you wish to keep (e.g. the
  weights of a network you have used in a paper).
* The ``data1`` folder should only contain **compressed** datasets.
* ``$SLURM_TMPDIR`` points to the local disk of the node on which a job is
  running. It should be used to copy the data on the node at the beginning of
  the job and write intermediate checkpoints. This folder is cleared after each
  job.

**Auto-cleanup** is applied on files not touched during the specified period

.. warning:: We currently do not have a backup system in the lab. You should backup whatever you want to keep on your laptop, google drive, etc.

