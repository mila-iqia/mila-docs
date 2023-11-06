.. _logging_in:

Logging in to the cluster
=========================

To access the Mila Cluster clusters, you will need a Mila account. Please contact
Mila systems administrators if you don't have it already. Our IT support service
is available here: https://it-support.mila.quebec/

You will also need to complete and return an IT Onboarding Training to get
access to the cluster.  Please refer to the Mila Intranet for more
informations:
https://sites.google.com/mila.quebec/mila-intranet/it-infrastructure/it-onboarding-training

**IMPORTANT** : Your access to the Cluster is granted based on your status at
Mila (for students, your status is the same as your main supervisor' status),
and on the duration of your stay, set during the creation of your account. The
following have access to the cluster : **Current Students of Core Professors -
Core Professors - Staff**



SSH Login
---------

You can access the Mila cluster via ssh:

.. prompt:: bash $

    # Generic login, will send you to one of the 4 login nodes to spread the load
    ssh <user>@login.server.mila.quebec -p 2222

    # To connect to a specific login node, X in [1, 2, 3, 4]
    ssh <user>@login-X.login.server.mila.quebec -p 2222

Four login nodes are available and accessible behind a load balancer. At each
connection, you will be redirected to the least loaded login-node.

The ECDSA, RSA and ED25519 fingerprints for Mila's login nodes are:

.. code-block:: text

    SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8 (ECDSA)
    SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o (RSA)
    SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg (ED25519)


.. important::
    Login nodes are merely *entry points* to the cluster. They give you access
    to the compute nodes and to the filesystem, but they are not meant to run
    anything heavy. Do **not** run compute-heavy programs on these nodes,
    because in doing so you could bring them down, impeding cluster access for
    everyone.

    This means no training or experiments, no compiling programs, no Python
    scripts, but also no ``zip`` of a large folder or anything that demands a
    sustained amount of computation.

    **Rule of thumb:** never run a program that takes more than a few seconds on
    a login node.

    .. note::
        In a similar vein, you should not run VSCode remote SSH instances directly
        on login nodes, because even though they are typically not very
        computationally expensive, when many people do it, they add up! See
        :ref:`Visual Studio Code` for specific instructions.


.. _mila_init:

mila init
---------

To make it easier to set up a productive environment, Mila publishes the
milatools_ package, which defines a ``mila init`` command which will
automatically perform some of the below steps for you. You can install it with
``pip`` and use it, provided your Python version is at least 3.8:

.. code-block:: bash

    $ pip install milatools
    $ mila init

.. _milatools: https://github.com/mila-iqia/milatools

.. note::
    This guide is current for ``milatools >= 0.0.17``. If you have installed an older
    version previously, run ``pip install -U milatools`` to upgrade and re-run
    ``mila init`` in order to apply new features or bug fixes.


SSH Config
----------

The login nodes support the following authentication mechanisms:
``publickey,keyboard-interactive``.  If you would like to set an entry in your
``.ssh/config`` file, please use the following recommendation:

.. code-block:: bash

   Host mila
       User YOUR-USERNAME
       Hostname login.server.mila.quebec
       PreferredAuthentications publickey,keyboard-interactive
       Port 2222
       ServerAliveInterval 120
       ServerAliveCountMax 5

Then you can simply write ``ssh mila`` to connect to a login node. You will also
be able to use ``mila`` with ``scp``, ``rsync`` and other such programs.

.. tip::
    You can run commands on the login node with ``ssh`` directly, for example
    ``ssh mila squeue -u '$USER'`` (remember to put single quotes around any
    ``$VARIABLE`` you want to evaluate on the remote side, otherwise it will be
    evaluated locally before ssh is even executed).


Passwordless login
------------------

To save you some repetitive typing it is highly recommended to set up public
key authentication, which means you won't have to enter your password every time
you connect to the cluster.

.. code-block:: bash

    # ON YOUR LOCAL MACHINE
    # You might already have done this in the past, but if you haven't:
    ssh-keygen  # Press ENTER 3x

    # Copy your public key over to the cluster
    # You will need to enter your password
    ssh-copy-id mila


Connecting to compute nodes
---------------------------

If (and only if) you have a job running on compute node "cnode", you are
allowed to SSH to it directly, if for some reason you need a second terminal.
That session will be automatically ended when your job is relinquished.

First, however, you need to have
password-less ssh either with a key present in your home or with an
``ssh-agent``. To generate a key pair on the login node:

.. code-block:: bash

    # ON A LOGIN NODE
    ssh-keygen  # Press ENTER 3x
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    chmod 700 ~/.ssh

Then from the login node you can write ``ssh <node>``. From your local
machine, you can use ``ssh -J mila USERNAME@<node>`` (-J represents a "jump"
through the login node, necessary because the compute nodes are behind a
firewall).

If you wish, you may also add the following wildcard rule in your ``.ssh/config``:

.. code-block::

    Host *.server.mila.quebec !*login.server.mila.quebec
        HostName %h
        User YOUR-USERNAME
        ProxyJump mila

This will let you connect to a compute node with ``ssh <node>.server.mila.quebec``.


Auto-allocation with mila-cpu
-----------------------------

If you install milatools_ and run ``mila init``, then you can automatically allocate
a CPU on a compute node and connect to it by running:

.. code-block:: bash

    ssh mila-cpu

And that's it! Multiple connections to ``mila-cpu`` will all reuse the same job, so
you can use it liberally. It also works transparently with VSCode's Remote SSH feature.

We recommend using this for light work that is too heavy for a login node but does not
require a lot of resources: editing via VSCode, building conda environments, tests, etc.

The ``mila-cpu`` entry should be in your ``.ssh/config``. Changes are at your own risk.
While it is possible to tweak it to allocate a GPU, doing so will prevent simultaneous
connections to it (until Slurm is upgraded to version 22.05 or later).
