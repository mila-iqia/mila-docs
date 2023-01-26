.. _quick_start:

Quick Start
===========

Users first need :ref:`login access to the cluster <logging_in>`. It is
recommended to install milatools_ which will help in the :ref:`set up of the
ssh configuration <mila_init>` needed to securely and easily connect to the
cluster.

.. _mila_code:

mila code
---------

milatools_ also makes it easy to run and debug code on the Mila cluster. Using
the ``mila code`` command will allow you to use `VSCode
<https://code.visualstudio.com/>`_ on the server. Simply run:

.. code-block:: bash

   mila code path/on/cluster

The details of the command can be found in the `mila code section of github page
<https://github.com/mila-iqia/milatools#mila-code>`_. Note that you need to
first setup your ssh configuration using ``mila init`` before the ``mila code``
command can be used. The initialisation of the ssh configuration is explained
:ref:`here <mila_init>` and in the `mila init section of github page
<https://github.com/mila-iqia/milatools#mila-init>`_.

.. _milatools: https://github.com/mila-iqia/milatools
