Portability concerns and solutions
==================================

Creating a list of your software's dependencies
-----------------------------------------------

TODO


Managing your envs
------------------

.. include:: Userguide_python.rst


Using Modules
-------------

Many software, such as Python and Conda, are already compiled and available on the cluster through the ``module`` command and
its sub-commands. In particular, if you with to use ``Python 3.7`` you can simply do:

.. prompt:: bash $

    module load python/3.7

.. include:: Userguide_portability_modules.rst


On using containers
-------------------

Another option for portable code might also be :ref:`Using containers`.

One popular mechanism for containerisation on a computational cluster is called
`singularity`. This is the recommended approach for running containers on the
Mila cluster.


.. include:: singularity/index.rst
