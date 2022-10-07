Software on the cluster
***********************

This section aims to raise awareness to problems one can encounter when trying
to run a software on different computers and how this is dealt with on typical
computation clusters.

The Mila cluster and the Digital Research Alliance of Canada clusters both
provide various useful software and computing environments, which can be
activated through the module system. Alternatively, you may build containers
with your desired software and run them on compute nodes.

Regarding Python development, we recommend using virtual environments to install
Python packages in isolation.


Cluster software modules
========================

Modules are small files which modify your environment variables to point to
specific versions of various software and libraries. For instance, a module
might provide the ``python`` command to point to Python 3.7, another might
activate CUDA version 11.0, another might provide the ``torch`` package, and so
on.

For more information, see :ref:`The module command`.


Containers
==========

Containers are a special form of isolation of software and its dependencies. A
container is essentially a lightweight virtual machine: it encapsulates a
virtual file system for a full OS installation, as well as a separate network
and execution environment.

For example, you can create an Ubuntu container in which you install various
packages using ``apt``, modify settings as you would as a root user, and so on,
but without interfering with your main installation. Once built, a container can
be run on any compatible system.

For more information, see :ref:`Using containers`.


Python Virtual environments
===========================

A virtual environment in Python is a local, isolated environment in which you
can install or uninstall Python packages without interfering with the global
environment (or other virtual environments). In order to use a virtual
environment, you first have to activate it.

For more information, see :ref:`Virtual environments`.
