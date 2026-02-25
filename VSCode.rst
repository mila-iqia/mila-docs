
Visual Studio Code
==================

One editor of choice for many researchers is VSCode. One feature of VSCode is
remote editing through SSH. This allows you to edit files on the cluster as if
they were local. You can also debug your programs using VSCode's debugger, open
terminal sessions, etc.


Connecting to the cluster
-------------------------

VSCode cannot be used to edit code on the login nodes, because it is a heavy
enough process (a ``node`` process, plus the language server, linter, and
possibly other plugins depending on your configured environment) that there is a
risk of overloading the login nodes if too many researchers did it at the same
time.

Therefore, to use VSCode on the cluster, you first need to allocate a compute
node, then connect to that node.

The simplest way to do so is to install milatools_ on your local machine and run the
``mila init`` command. This will add a ``mila-cpu`` entry to your SSH configuration
that automatically allocates a CPU on a compute node for you. Then you can simply
choose ``mila-cpu`` in the dropdown menu for remote SSH connection.

The milatools_ package also provides the :ref:`mila code command <mila_code>` which
gives you greater flexibility by letting you allocate more cores, RAM and/or GPUs.

.. _milatools: https://github.com/mila-iqia/milatools

Activating an environment
-------------------------

Reference_

.. _reference: https://code.visualstudio.com/docs/python/environments

To activate a conda or pip environment, you can open the command palette with
Ctrl+Shift+P and type "Python: Select interpreter". This will prompt you for the
path to the Python executable for your environment.

.. tip::

    If you already have the environment activated in a terminal session, you can
    run the command ``which python`` to get the path for this environment. This
    path can be pasted into the interpreter selection prompt in VSCode to use
    that same environment.


Troubleshooting
---------------

"Cannot reconnect"
^^^^^^^^^^^^^^^^^^

When connecting to multiple compute nodes (and/or from multiple computers), some
instances may crash with that message because of conflicts in the lock files
VSCode installs in ``~/.vscode-server`` (which is shared on all compute nodes).
To fix this issue, you can change this setting in your ``settings.json`` file:

.. code-block:: json

   { "remote.SSH.lockfilesInTmp": true }

This will store the necessary lockfiles in ``/tmp`` on the compute nodes (which
are local to the node).


Debugger timeouts
^^^^^^^^^^^^^^^^^

Sometimes, slowness on the compute node or the networked filesystem might cause
the VSCode debugger to timeout when starting a remote debug process. As a quick
fix, you can add this to your ``~/.bashrc`` or ``~/.profile`` or equivalent
resource file for your preferred shell, to increase the timeout delay to 500
seconds:

.. code-block:: bash

    export DEBUGPY_PROCESS_SPAWN_TIMEOUT=500
