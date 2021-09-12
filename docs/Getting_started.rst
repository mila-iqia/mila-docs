

.. It seems this section would need to be broken up into the current existing
  sections (cluster theory and / or user_guide), hence I am not reactivating this
  section just yet.



Getting started
===============

**Before starting:** get your Mila cluster account ready. Typically, your
username is identical to your Mila account (the part before ``@mila.quebec``).
In some special cases, it may look like the first seven letters of your last
name and the first letter of your first name, but this may vary.


Connect to the cluster
----------------------

First, make sure you have an ``ssh`` client installed. There should be one
installed by default on Linux, MacOS or Windows. Then you can connect to a login
node as follows:


.. code-block:: bash

    # Generic login, will send you to one of the 4 login nodes to spread the load
    ssh user@login.server.mila.quebec -p 2222

    # To connect to a specific login node, X in [1, 2, 3, 4]
    ssh user@login-X.login.server.mila.quebec -p 2222


To save some typing (highly recommended), you can include the information in
your ssh config file. For instance, you can add the following lines in
``.ssh/config`` on your local machine:

.. code-block::

    Host mila
        User YOUR-USERNAME
        HostName login.server.mila.quebec
        Port 2222

Then you can simply write ``ssh mila`` to connect to a login node. You will also
be able to use ``mila`` with ``scp``, ``rsync`` and other such programs. For the
rest of this page we will assume that ``.ssh/config`` contains the entry above.

To save even more typing (again, *highly recommended*), you should set up public
key authentication, which means you won't have to enter your password every time
you connect to the cluster.

.. code-block:: bash

    # ON YOUR LOCAL MACHINE
    # You might already have done this in the past, but if you haven't:
    ssh-keygen  # Press ENTER 3x

    # Copy your public key over to the cluster
    ssh-copy-id mila


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


Compute nodes
-------------

Your computations should be performed on *compute nodes*. They are accessed
through an allocation system called SLURM, which manages requests from the
cluster's hundreds of users. You can:

* Request an interactive job with ``salloc``, which lets you freely play around
* Run a single job using ``srun``
* Run a set of jobs using ``sbatch`` (for hyperparameter searches and the like)


Interactive job
---------------

Of course, the first thing you'll want to do is mess around with the cluster's
software environment. To connect interactively to a compute node, use
``salloc``. Note that it will not connect you immediately and it might take a
few minutes.


.. code-block:: bash

    # Basic allocation, no GPU
    salloc

    # Allocation with one GPU
    salloc --gres=gpu:1

    # Allocation with one GPU and 10G of RAM
    salloc --gres=gpu:1 --mem=10G


When your interactive job is allocated, you will be dropped into a shell on a
compute node. The job ends when you quit the shell.

.. tip::
    If you have a job running on compute node "cnode", you are allowed to SSH to
    it directly, if for some reason you need a second terminal. First you will
    need to generate a key pair on the login node (``ssh-keygen``) and then add
    that to the authorized keys on the same drive: ``cat ~/.ssh/id_rsa.pub >>
    ~/.ssh/authorized_keys``.

    Then from the login node you can write ``ssh cnode``. From your local
    machine, you can use ``ssh -J mila USERNAME@cnode`` (-J represents a "jump"
    through the login node, necessary because the compute nodes are behind a
    firewall). When the main session is terminated, the other connections are
    also terminated.


Getting stuff to/from the cluster
---------------------------------

Now for a small interlude: how do you get your code and data on the cluster? How
do you transfer your data back? There are many ways to do so, but you should at
least familiarize yourself with the ``scp`` and ``rsync`` commands to transfer
data from and to your local machine, ``wget`` or ``curl`` to download data from
the Internet, and of course ``git``.

For now let's suppose you have a local directory called ``myproject`` that you
want to send to the cluster:

.. code-block:: bash

    # This will copy myproject to $HOME/myproject on the cluster
    rsync -av myproject mila:

    # Same as above. Note and remember how the trailing slash on myproject/
    # changes the meaning of the command.
    rsync -av myproject/ mila:myproject


Running a script
----------------

.. important::
    **REMINDER**: do not run any of the commands below on the login node. Use
    ``salloc`` first to get a session on a compute node.

For simplicity, let's download an example script from the Internet:

.. code-block:: bash

    wget TODO/example.py

That script uses PyTorch. The cluster comes with appropriate modules to use it,
for instance in this case we can load miniconda and then pytorch 1.7:

.. code-block:: bash

    module load miniconda/3 pytorch/1.7

.. note::
    The complete list of available modules can be seen with ``module avail``.
    Some modules depend on other modules, they will either be loaded
    automatically or the module command will tell you.

Then you can simply run it:

.. code-block:: bash

    python example.py

If you need to install extra Python packages that are not included in the
module, or if you need e.g. the latest version of PyTorch and it's not yet
available as a module, you will need to create a virtual environment (See
:ref:`Virtual environments` for more information). You can also build and use
Docker or Singularity containers. See :ref:`Using containers` for more
information.


Batch job
---------

Once you are satisfied with your script and want to run a large number of
experiments (or run an experiment non-interactively), you will need to create a
shell script that can be given to the ``sbatch`` command.

For example, write this in ``example-batch.sh`` (tip: write ``cat >
example-batch.sh`` in the terminal, paste the code, and hit Ctrl+D to save it):

.. code-block:: bash

    #!/bin/bash

    # sbatch will read any lines that start with "#SBATCH" and will
    # add what follows as if they were command line parameters.

    #SBATCH --job-name=example
    #SBATCH --output=job_output.txt
    #SBATCH --error=job_error.txt
    #SBATCH --ntasks=1
    #SBATCH --time=10:00
    #SBATCH --mem=100Gb

    # Load the necessary modules
    module load miniconda/3 pytorch/1.7

    # Activate a virtual environment, if needed:
    # conda activate myenv

    # Run the script
    python example.py

To launch the experiments, simply run ``sbatch`` on a login node:

.. code-block:: bash

    sbatch example-batch.sh

This will request a resource allocation for each task, queue these requests, and
once the jobs start running (be patient), the stdout/stderr will be stored in
files with the specified names.

Check on the status of your jobs with:

.. code-block:: bash

    squeue -u $USER

.. tip::
    You can run commands on the login node with ``ssh`` directly, for example
    ``ssh mila squeue -u '$USER'`` (remember to put single quotes around any
    ``$VARIABLE`` you want to evaluate on the remote side, otherwise it will be
    evaluated locally before ssh is even executed).


.. Best practices
   --------------

   TODO: Links to how to handle datasets, how to create Python projects, etc.


   Other resources
   ---------------

   TODO: Links to whatever may be useful to beginners
