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


.. _SSH:

SSH (Secure Shell)
------------------

**All access to the Mila cluster is via SSH using public-key authentication.**
As of **March 31, 2025**, this will become the **only** means of authentication,
and **password-based authentication will no longer work**.

SSH key authentication is a technique using pairs of closely-linked keys: A
private key, and a corresponding public key. The public key should be
distributed to everyone, while the private key is known to only one person. The
public key can be used by anyone to challenge a person to prove their identity.
If they have the corresponding private key, that person can perform an
electronic signature that everyone can validate but that no one else could have
done themselves. The challenge is thus answered by demonstrating possession of
the private key (and therefore their identity), without ever revealing the
private key itself.

*Mila asks you to generate a pair of SSH keys, to provide Mila only with your
public key,* which has no confidentiality implications, and to keep the private
key for yourself. *The private key must remain secret and solely known to you,*
because anyone who possesses it is capable of impersonating you by performing
your electronic signature.

During the IT Onboarding Training, you will be asked to submit that SSH
public key.

- If you do not know what SSH keys are, or are not familiar with them, you can
  read the informative material :ref:`below<SSH Private Keys>`, then proceed to
  generate them.
- If you do not already have SSH keys, or are not sure if you have them, skip
  to the instructions on how to generate them :ref:`here<checking_for_ssh_keys>`.
- If you do have SSH keys, you can skip to :ref:`configuring SSH for access to Mila<Configuring SSH>`.


Logging in with SSH
^^^^^^^^^^^^^^^^^^^

Login to the Mila cluster is with ``ssh`` through four Internet-facing
*login nodes* and a load-balancer. At each connection through the load-balancer,
you will be redirected to the least loaded login node.

.. prompt:: bash $

    # Generic login, will send you to one of the 4 login nodes to spread the load
    ssh -p 2222 <user>@login.server.mila.quebec

    # To connect to a specific login node, X in [1, 2, 3, 4]
    ssh -p 2222 <user>@login-X.login.server.mila.quebec

This is a significant amount of typing. You are **strongly** encouraged to add
a ``mila`` "alias" to your SSH configuration file (see :ref:`below<Configuring SSH>`
for how). With a correctly-configured SSH you can now simply run

.. prompt:: bash $

    # Login with SSH configuration in place
    ssh mila

    # Can also scp...        vvvv
    scp  file-to-upload.zip  mila:scratch/uploaded.zip

    #          vvvv  ... and rsync!
    rsync -avz mila:my/remote/sourcecode/  downloaded-source/

to connect to a login node. The ``mila`` alias will be available to ``ssh``,
``scp``, ``rsync`` and all other programs that consult the SSH configuration file.

Upon first login, you may be asked to enter your SSH key passphrase. Use the
passphrase you used to create your SSH key :ref:`below<generating_ssh_keys>`.

Upon first login, you may also be asked whether you trust the ***Mila*** login
servers' *own* SSH keys.
The ECDSA, RSA and ED25519 fingerprints for Mila's login nodes are:

.. code-block:: text

    SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8 (ECDSA)
    SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o (RSA)
    SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg (ED25519)

If the fingerprints presented to you do not match one of the above, **do not**
trust them!

.. tip::
    You can run commands on the login node with ``ssh`` directly, for example
    ``ssh mila squeue -u '$USER'`` (remember to put single quotes around any
    ``$VARIABLE`` you want to evaluate on the remote side, otherwise it will be
    evaluated locally before ssh is even executed).


.. important::
    Login nodes are merely *entry points* to the cluster. They give you access
    to the compute nodes and to the filesystem, but they are not meant to run
    anything heavy. Do **not** run compute-heavy programs on these nodes,
    because in doing so you could bring them down, impeding cluster access for
    everyone.

    This means no training scripts or experiments and no compilation of software
    unless it is small or ends quickly. Do not run anything that demands a
    sustained large amount of computation or a large amount of memory.

    **Rule of thumb:** Never run a program that takes more than a few seconds on
    a login node, unless it mostly sleeps or mostly moves data.

    **Examples:** A non-exhaustive list of use-cases, to give a sense of what is
    and is not allowed on the login nodes:

    - A Python training script is unacceptable on the login nodes.
      *(Too computationally- and memory-intensive)*
    - A Python or shell script that downloads a dataset and exits immediately
      after may be acceptable on the login nodes.
      *(Mostly moves data)*
    - A Python hyperparameter search script that uses ``submitit`` to launch
      jobs and only sleeps waiting for them to end and run other jobs is
      acceptable on the login nodes.
      *(Mostly sleeps; The actual jobs run on the compute nodes)*
    - ``pip install`` of ``vllm`` or ``flash-attn`` from source code on the
      login nodes is unacceptable (and is likely to fail anyways).
      *(Takes far too much RAM to compile the CUDA kernels)*
    - Editing code with ``nano``, ``vim`` or ``emacs`` is acceptable.
      *(Editors mostly sleep awaiting user keystrokes)*
    - Copying/moving files with ``cp``, ``mv``, ... is acceptable.
      *(Mostly moves data)*
    - Connecting to compute nodes with ``ssh`` is acceptable.
      *(Mostly sleeps, forwarding keystrokes and ports to/from the node)*
    - Using ``tmux`` is acceptable.
      *(Mostly sleeps, managing the processes under its control)*

    .. note::
        In a similar vein, you should not run VSCode remote SSH instances directly
        on login nodes, because even though they are typically not very
        computationally expensive, when many people do it, they add up! See
        :ref:`Visual Studio Code` for specific instructions.





SSH Private Keys
^^^^^^^^^^^^^^^^

A private SSH key commonly takes the form of an obscure text file. It encodes
the digital secret of how to make an electronic signature — specifically, yours.
The content of a private SSH key might resemble

.. code-block:: text

    -----BEGIN OPENSSH PRIVATE KEY-----
    b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
    NhAAAAAwEAAQAAAYEAl5dD/UU2CvauaVS2/4/iWoUyO1Hey+m8KojCFMvIywL6PPdYRqVa
    FOidmOw/E9V2HVzHz/z/2Dj6TO5xNX1qJFk7A/ACGGc1+KguIDQWdjR6AZb5Tat+aAMYro
    …
    aSeJOS59knbQJeBwPm0g5G+iFz6R17446dXk5jn3/29AutF5MPnKwqE0mjywxCLYxVX3He
    YSOCZfE80P/z4sImW82BYxAzKtI8kKagLmHS4gXJEmE13Dfyq0xcB3q5OMuQ2fZwvukTx3
    xdWgyqFrMyC4wHAAAAAAEC
    -----END OPENSSH PRIVATE KEY-----

In the real world, a handwritten signature is useless for authenticating you if
it can be easily reproduced by others. In the virtual world, the same is true.
Anyone who has your private key is capable of reproducing your electronic signature.
It is therefore essential that only one person — you — holds this private key.
The secrecy of the private key is the guarantor of your online identity.

Mila will ***never*** ask you for your private SSH key, and any pretense of
request for a private key constitutes an attempt at phishing and identity theft.
Keep your private keys safe and do not share them with anyone. Do not put them
in the cloud, your emails, Slack messages, or Git repos. Protect them with a
passphrase.


SSH Public Keys
^^^^^^^^^^^^^^^

A public SSH key is a simple line of text, albeit sometimes very long, commonly
found in a file with the ``.pub`` extension. It encodes the digital knowledge
required to recognize and validate your electronic signature, without however
making it possible to reproduce it elsewhere. Here are three examples of public
SSH keys:

.. code-block:: text

    ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDMYpSndal/…mPL+NXs=
    ssh-ed25519 AAAA…d/ca2h  user@server
    ecdsa-sha2-nistp256 AAAA…hWQcQg8=  mylaptop

You are requested to submit just such a public SSH key to Mila, which will
allow Mila to recognize you when you connect to the Mila cluster, but without
revealing the secret of how to perform your signature.


.. _checking_for_ssh_keys:

Checking If You Already Have SSH (Private) Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually, a private SSH key is found in the hidden directory ``~/.ssh/`` and is
named ``id_rsa``, ``id_ed25519``, or ``id_ecdsa``. Its corresponding public SSH
key is usually in the same directory and shares the same filename, except with
a ``.pub`` suffix.


.. _generating_ssh_keys:

Generating an SSH Private Key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If no private SSH key already exists, you can create one with the ``ssh-keygen`` utility:

+--------------------------------------+-------------------------------+
|                 RSA                  |            Ed25519            |
+--------------------------------------+-------------------------------+
| *Integer factorization*              | *Elliptic curve*              |
+======================================+===============================+
| - Classic                            | - New                         |
| - Ultra-compatible, standardized,    | - Less compatible             |
|   the reference                      | - Fixed, small key size       |
| - Large key size, but configurable   | - Fast                        |
| - Slow or even very slow             |                               |
+--------------------------------------+-------------------------------+
| ``$ ssh-keygen -t rsa -b 3072``      | ``$ ssh-keygen -t ed25519``   |
|                                      |                               |
| ``(enter passphrase)``               | ``(enter passphrase)``        |
|                                      |                               |
| ``(re-enter passphrase)``            | ``(re-enter passphrase)``     |
+--------------------------------------+-------------------------------+

.. tip::
    The pass-**phrase** protects the SSH private key **on-disk**. The
    passphrase is **not** the same thing as the pass-**word** used to *log into
    your personal computer account*. However, choosing them to be equal may
    allow for automatic unlocking of encrypted SSH private keys at login, in
    combination with software such as ``pam_ssh(8)`` (Linux) or Keychain
    (Mac OS X/macOS). This makes the good practice of using encrypted keys
    convenient as well.


Generating an SSH Public Key from a Private Key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a private SSH key exists, but not its corresponding SSH public key, it can
be recalculated with the ``ssh-keygen`` utility as well:

+--------------------------------------+------------------------------------------+
|                 RSA                  |            Ed25519                       |
+======================================+==========================================+
| **SSH public key:**                  | **SSH public key:**                      |
|                                      |                                          |
| >380 bytes @ 2048 bits (not rec.)    | ~82 bytes                                |
|                                      |                                          |
| >550 bytes @ 3072 bits (recommended) |                                          |
|                                      |                                          |
| >725 bytes @ 4096 bits (slower)      |                                          |
|                                      |                                          |
| >1400 bytes @ 8192 bits (much slower)|                                          |
+--------------------------------------+------------------------------------------+
| ``$ ssh-keygen -y -f ~/.ssh/id_rsa`` | ``$ ssh-keygen -y -f ~/.ssh/id_ed25519`` |
|                                      |                                          |
| ``(enter passphrase)``               | ``(enter passphrase)``                   |
+--------------------------------------+------------------------------------------+

It is this SSH public key that you should submit in the IT Onboarding Training form.



Configuring SSH
---------------

SSH uses a configuration file ``~/.ssh/config`` (right next to the SSH keys)
to indicate which connection settings to use for each SSH server one can
connect to.

The Mila **login** nodes require:

- ``Hostname``: ``login.server.mila.quebec``
- ``Port``: ``2222``
- ``User``: *Your Mila account username*
- ``PreferredAuthentications``: ``publickey,keyboard-interactive``

Password authentication will be withdrawn on :ref:`March 31, 2025<SSH>`.

A simple SSH configuration is automatically created and added for you to
``~/.ssh/config`` by :ref:`mila init`.

Alternatively, more advanced users can edit the SSH ``.config`` file
:ref:`manually<manual_ssh_config>`.


.. _manual_ssh_config:

Manual SSH configuration
^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to set entries in your ``~/.ssh/config`` file manually for
advanced use-cases, you may use the following as inspiration:

.. code-block:: text

    #   Mila
    Host mila             login.server.mila.quebec
        Hostname          login.server.mila.quebec
    Host mila1    login-1.login.server.mila.quebec
        Hostname  login-1.login.server.mila.quebec
    Host mila2    login-2.login.server.mila.quebec
        Hostname  login-2.login.server.mila.quebec
    Host mila3    login-3.login.server.mila.quebec
        Hostname  login-3.login.server.mila.quebec
    Host mila4    login-4.login.server.mila.quebec
        Hostname  login-4.login.server.mila.quebec
    Host mila5    login-5.login.server.mila.quebec
        Hostname  login-5.login.server.mila.quebec
    Host cn-????
        Hostname             %h.server.mila.quebec
    Match host *.server.mila.quebec !*login.server.mila.quebec
        Hostname                 %h
        ProxyJump                mila
    Match host           *login.server.mila.quebec
        Port                     2222
        ServerAliveInterval      120
        ServerAliveCountMax      5
    Match host *.server.mila.quebec
        PreferredAuthentications publickey,keyboard-interactive
        AddKeysToAgent           yes
        ## Consider uncommenting:
        # ForwardAgent             yes
        ## Delete if on Linux, uncomment if on Mac:
        # UseKeychain              yes
        User                     CHANGEME_YOUR_MILA_USERNAME

.. important::
    Please make the required edits to the template above, especially regarding
    ``CHANGEME_YOUR_MILA_USERNAME``!


.. _mila_init:

mila init
^^^^^^^^^

To make it easier to set up a productive environment, Mila publishes the
milatools_ package, which defines a ``mila init`` command which will
automatically perform some of the below steps for you. You can install it with
``pip`` and use it, provided your Python version is at least 3.9:

.. prompt:: bash $

    pip install milatools
    mila init

.. _milatools: https://github.com/mila-iqia/milatools

.. note::
    This guide is current for ``milatools >= 0.0.17``. If you have installed an older
    version previously, run ``pip install -U milatools`` to upgrade and re-run
    ``mila init`` in order to apply new features or bug fixes.


Connecting to compute nodes
---------------------------

If (and only if) you have a job running on compute node ``cnode``, you are
allowed to SSH to it, if for some reason you need a second terminal.
That session will be automatically ended when your job ends.

First, however, you need to have
password-less ssh either with a key present in your home or with an
``ssh-agent``. To generate a key pair on the login node:

.. prompt:: bash $

    # ON A LOGIN NODE
    ssh-keygen  # Press ENTER 3x
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    chmod 700 ~/.ssh

Then from the login node you can write ``ssh cnode``. From your local
machine, you can use ``ssh -J mila USERNAME@cnode`` (``-J`` represents a "jump"
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

.. prompt:: bash $

    ssh mila-cpu

And that's it! Multiple connections to ``mila-cpu`` will all reuse the same job, so
you can use it liberally. It also works transparently with VSCode's Remote SSH feature.

We recommend using this for light work that is too heavy for a login node but does not
require a lot of resources: editing via VSCode, building conda environments, tests, etc.

The ``mila-cpu`` entry should be in your ``.ssh/config``. Changes are at your own risk.


Using a non-Bash Unix shell
---------------------------

While Mila does not provide support in debugging your shell setup, Bash is the
standard shell to be used on the cluster and the cluster is designed to support
both Bash and Zsh shells. If you think things should work with Zsh and they
don't, please contact `Mila's IT support <https://it-support.mila.quebec>`_.
