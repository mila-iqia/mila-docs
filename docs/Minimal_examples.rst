****************
Minimal Examples
****************

This section contains some minimal examples of how to run jobs on the Mila cluster.

Each example is self-contained and can be run as-is directly on the cluster without error.
Examples has the following structured:

* ``job.sh``: SLURM ``sbatch`` script. Can be launched with ``sbatch job.sh``.
* ``main.py``: Example python script.

Some examples (for example :doc:`this one <examples/distributed/multi_gpu/_index>`) are displayed in terms of the difference
between some "base" example and the "new" example.



.. toctree::
    :maxdepth: 1

    examples/frameworks/index
    examples/distributed/index
    examples/good_practices/index
