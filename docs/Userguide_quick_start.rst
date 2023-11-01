.. _quick_start:

Quick Start
===========

Users first need :ref:`login access to the cluster <logging_in>`. It is
recommended to install milatools_ which will help in the :ref:`set up of the
ssh configuration <mila_init>` needed to securely and easily connect to the
cluster.

.. _mila_code:

Windows
-------

If you are using Windows, here is how you should go about setting up access to the cluster:

* Install the Windows Subsystem for Linux
   1. Open a command prompt (cmd)
   2. Enter the following command and follow the prompts, choosing a username and password, etc.
   
         .. code-block:: console

            $ wsl --install -d ubuntu
   
   If you encounter any difficulties, check out `the official guide for installing the Windows
   Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ 

* Inside the WSL (ubuntu) shell, install Anaconda:

   .. code-block:: console

      $ # This will download the Anaconda installer.
      $ wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
      $ bash Anaconda3-2023.09-0-Linux-x86_64.sh
   
   * This will run the installer. Carefully press <Enter> until you see the prompt to accept 
     the license agreement. Write "yes" and press enter.
   * When asked where to install Anaconda, press <Enter> to accept the default location.


* Activate the base environment and install the milatools_ python package:

   .. code-block:: console

      $ conda activate
      $ pip install milatools

* Run ``mila init`` to setup your SSH configuration.

   .. code-block:: console

      $ pip install milatools
      $ mila init

* Install `Visual Studio Code <https://code.visualstudio.com/>`_ and the `Remote-WSL extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`_
   * Make sure that you're able to execute the ``code`` command within the WSL shell.


.. * Download and install `Anaconda <https://www.anaconda.com/download#downloads>`_
.. might also be relevant:
.. https://code.visualstudio.com/docs/remote/wsl-tutorial

mila code
---------

milatools_ also makes it easy to run and debug code on the Mila cluster.

First you need to setup your ssh configuration using ``mila init``. The
initialisation of the ssh configuration is explained
:ref:`here <mila_init>` and in the `mila init section of github page
<https://github.com/mila-iqia/milatools#mila-init>`_.

Once that is done, you may run `VSCode <https://code.visualstudio.com/>`_
on the cluster simply by `using the Remote-SSH extension <https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host>`_
and selecting ``mila-cpu`` as the host (in step 2).

``mila-cpu`` allocates a single CPU and 8 GB of RAM. If you need more
resources from within VSCode (e.g. to run a ML model in a notebook), then
you can use ``mila code``. For example, if you want a GPU, 32G of RAM and 4 cores,
run this command in the terminal:

.. code-block:: bash

   mila code path/on/cluster --alloc --gres=gpu:1 --mem=32G -c 4

The details of the command can be found in the `mila code section of github page
<https://github.com/mila-iqia/milatools#mila-code>`_. Remember that you need to
first setup your ssh configuration using ``mila init`` before the ``mila code``
command can be used.

.. _milatools: https://github.com/mila-iqia/milatools
