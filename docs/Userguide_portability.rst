Portability concerns and solutions
==================================


Creating a list of your software's dependencies
-----------------------------------------------

TODO


Managing your environments
--------------------------

.. include:: Userguide_python.rst


Using Modules
-------------

Many software, such as Python and Conda, are already compiled and available on
the cluster through the ``module`` command and its sub-commands. In particular,
if you with to use ``Python 3.7`` you can simply do:

.. prompt:: bash $

    module load python/3.7

.. include:: Userguide_portability_modules.rst


On using containers
-------------------

Containers are a popular approach at deploying applications
by packaging a lot of the required dependencies together.

The most popular tool for this is Docker, but Docker cannot
be used on the Mila cluster (nor the other clusters from Compute Canada).

The alternative is to use `Singularity
<https://singularity-docs.readthedocs.io/en/latest/>`_.  See section
:ref:`Singularity` for more details.  This is the recommended solution for
running containers on the Mila cluster, it works very well, and it supports GPUs
(much like Docker does).

