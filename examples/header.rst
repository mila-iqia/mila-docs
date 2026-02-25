.. note::

    .. container:: toggle

        .. container:: header

            About these examples: (Click the arrow below to expand)

        This section contains some minimal examples of how to run jobs on the Mila
        cluster.
        Each example is self-contained and can be run as-is directly on the cluster
        without error.  Each example has the following structure:

        * ``job.sh``: SLURM ``sbatch`` script. Can be launched with ``sbatch job.sh``.
        * ``main.py``: Example python script.

        Some examples are displayed as a difference with respect to a "base" example.
        For instance, the :doc:`multi-gpu example
        </examples/distributed/multi_gpu/index>` is shown as a difference with respect
        to the :doc:`single-gpu example </examples/distributed/single_gpu/index>`.
