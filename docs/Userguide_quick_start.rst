.. _quick_start:

Quick Start
===========

.. include:: Userguide_cluster_access.rst


.. _mila_code:

Windows
-------

If you are using Windows, here is how you should go about setting up access to the cluster:
Note: we're assuming Windows 11, but this guide should also work for Windows 10 without too much
trouble.

.. * Install the Windows Subsystem for Linux
   
..    We recommend using the newest version of WSL, which is available for Windows 10 and Windows 11.
   
..    1. Open a command prompt (cmd)
..    2. Enter the following command to install WSL.

..          .. code-block:: console

..             $ wsl --install -d ubuntu

..    3. Restart your computer.

..    If you encounter any difficulties or want more information, please check out `the official
..    instructions for installing the Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ 

* Install `Visual Studio Code <https://code.visualstudio.com/>`_.
   * Install the Remote-SSH extension for VSCode
     This extension can be found at `this link <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh>`_.
     
     You can also install it from within VsCode: launch the VS Code Quick Open (Ctrl+P), paste the
     following command, and press enter:
     ``ext install ms-vscode-remote.remote-ssh``

* Install Miniconda:

   We recommend using Miniconda. It's a minimal version of Anaconda without as many pre-installed
   packages. This makes it smaller, meaning a faster download and less disk space used.

   .. code-block:: console

      $ # This will download the Anaconda installer.
      $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      $ bash Miniconda3-latest-Linux-x86_64.sh -b
      $ source ~/miniconda3/bin/activate
      $ conda init bash

   * This will run the installer and accept the license agreement.

   Alternatively, you can also use Anaconda:
   
   .. code-block:: console

      $ # This will download the Anaconda installer.
      $ wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
      $ bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
      $ conda init bash
   
   Make sure to close the shell and open a new one after running the above commands.

* In a Powershell window, install the milatools_ python package inside the "base" conda environment:
   Note: Assuming you ran ``conda init`` above, your new shell should now have the "base"
   environment activated.
   * If the ``conda`` command isn't found, you may need to add Anaconda to your Path environment
     variable. TODO: Until we write some instructions on how to do this, please search online for
     "add anaconda to path on windows" and try that.

   .. code-block:: console

      $ conda activate  # This is only required if you aren't already in the base environment.
      (base) $ pip install milatools


* Run ``mila init`` to setup a (part of) your SSH configuration.

   NOTE: This will NOT work completely! You should get an error involving the `ssh-copy-id` command
   not being found. This is a known issue that is being worked on at IDT.

   .. code-block:: console

      (base) $ mila init  # Enter all required information. You WILL get an error at the end.

* Add your public key to the authorized keys on the cluster:
   
   .. code-block:: console

      $ cat ~/.ssh/id_rsa.pub | ssh mila 'cat >> ~/.ssh/authorized_keys'
      password: # (enter your password)

* Logout, then try logging-in again
   If everything went well, you should not be prompted with a password.

* Launch `mila code .` from a Powershell window.
   This will open a VSCode window on the cluster. You can now edit files and run commands on the
   cluster from within VSCode.

If you're having any problems, please create an issue on the milatools GitHub repository.

mila code
---------

It is recommended to install milatools_ which will help in the :ref:`set up of
the ssh configuration <mila_init>` needed to securely and easily connect to the
cluster. milatools_ also makes it easy to run and debug code on the Mila
cluster.

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


Using a Terminal
----------------

While VSCode provides a graphical interface for writing and debugging code on
the cluster, working on the cluster will require to use a terminal to navigate
the filesystem, run commands, and manage jobs.

To open a terminal session on the cluster, connect using:

.. prompt:: bash $

   ssh mila

This will connect you to a login node where you can run commands, submit jobs,
and navigate the cluster filesystem.


Next Steps
----------

Once you have access to the cluster, you may want to:

* **Set up a framework**: For a quick example of setting up PyTorch on the
  cluster, see the :ref:`PyTorch Setup`.

* **Keep these references handy**:

  * The :ref:`Cheat Sheet` provides a quick reference for common commands and
    information about the Mila and DRAC clusters.

  * For a comprehensive reference of common terminal commands, see the
    `command line cheat sheet <https://cli-cheatsheet.readthedocs.io/>`_.

.. note::

   Before running a minimal example, make sure to read the :ref:`Running your
   code` guide, which explains how to submit jobs using Slurm and provides
   essential information about job submission arguments, partitions, and useful
   commands.
