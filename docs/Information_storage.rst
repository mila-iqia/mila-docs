.. _milacluster_storage:


Storage
=======

=========================================== =========== ====================================== =================== ============
Path                                        Performance Usage                                  Quota (Space/Files) Auto-cleanup
=========================================== =========== ====================================== =================== ============
``$HOME`` or ``/home/mila/<u>/<username>/`` Low         * Personal user space                  200G/1000K
                                                        * Specific libraries, code, binaries
``/network/projects/<groupname>/``          Fair        * Shared space to facilitate           200G/1000K
                                                          collaboration between researchers
                                                        * Long-term project storage
``/network/data1/``                         High        * Raw datasets (read only)
``/network/datasets/``                      High        * Curated raw datasets (read only)
``/miniscratch/``                           High        * Temporary job results                                    90 days
                                                        * Processed datasets
                                                        * Optimized for small Files
                                                        * Supports ACL to help share the
                                                          data with others
``$SLURM_TMPDIR``                           Highest     * High speed disk for temporary job    4T/-                at job end
                                                          results
=========================================== =========== ====================================== =================== ============

* ``$HOME`` is appropriate for codes and libraries which are small and read
  once, as well as the experimental results that would be needed at a later
  time (e.g. the weights of a network referenced in a paper).
* ``projects`` can be used for collaborative projects. It aims to ease the
  sharing of data between users working on a long-term project. It's possible
  to request a bigger quota if the project requires it.
* ``datasets`` contains curated datasets to the benefit of the Mila community.
  To request the addition of a dataset or a preprocessed dataset you think
  could benefit the research of others, you can fill `this form
  <https://forms.gle/vDVwD2rZBmYHENgZA>`_.
* ``data1`` should only contain **compressed** datasets. `Now deprecated and
  replaced by the` ``datasets`` `space.`
* ``miniscratch`` can be used to store processed datasets, work in progress
  datasets or temporary job results. Its blocksize is optimized for small files
  which minimizes the performance hit of working on extracted datasets. It
  supports ACL which can be used to share data between users. This space is
  cleared weekly and files older then 90 days will be deleted.
* ``$SLURM_TMPDIR`` points to the local disk of the node on which a job is
  running. It should be used to copy the data on the node at the beginning of
  the job and write intermediate checkpoints. This folder is cleared after each
  job.

.. note:: **Auto-cleanup** is applied on files not read or modified during the
   specified period

.. warning:: Currently there are no backup system in the lab. Storage local to
   personal computers, Google Drive and other related solutions should be used
   to backup important data
