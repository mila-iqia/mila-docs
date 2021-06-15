Software dependency management and associated challenges
========================================================

This section aims to raise awareness to problems one can encounter when trying
to run a software on different computers and how this is dealt with on typical
computation clusters.

Python Virtual environments
---------------------------

TODO

Cluster software modules
------------------------

Both Mila and Compute Canada clusters provides various software through the
``module`` command.  Modules are small files which modify your environment
variables to register the correct location of the software you wish to use. To
learn practical examples of module uses, see :ref:`The module command`.

Containers
----------

Containers are a special form of isolation of software and it's dependencies. It
does not only create a separate file system, but can also create a separate
network and execution environment. All software you have used for your
experiments is packaged inside one file. You simply copy the image of the
container you built on every environment without the need to install anything.
