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

milatools_ also helps using and debugging on the Mila cluster. Using the ``mila
code`` command will allow you to use `VSCode <https://code.visualstudio.com/>`_
on the server. Simply run:

.. code-block:: bash

   mila code path/on/cluster

The details of the command can be found on the `github page of the package
<https://github.com/mila-iqia/milatools#mila-code>`_. Note that you will first
need your ssh configuration to be done using the ``mila init`` command. The
initialisation of the ssh configuration is explained :ref:`here <mila_init>`
and on the `github page of the package
<https://github.com/mila-iqia/milatools#mila-init>`_.

.. _milatools: https://github.com/mila-iqia/milatools