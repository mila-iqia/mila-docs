
.. _home transition:

New server for ``/home/`` directories
=======================================

As things continue in these uncertain times, the new server to host your home directories
is ready. It's an all-flash server based on 48 NVMe disks and the BeeGFS parallel file system.

To ensure a smooth transition we intend on keeping both servers running for a while and the transition
will be on a voluntary basis. Before starting the process, remember to clean up your home directory
a bit to accelerate the transfer for you and others.


.. sectnum::


How to opt-in for the new server?
---------------------------------

Simply create a file called ``.flashme`` in your home directory

.. prompt:: bash $

    cd ~
    touch .flashme

A script a regularly scanning all home directories to gather users willing
to activate the transition.
Once your account is detected, the file will be renamed ``.flashme_OK``
and your account will be added to the following file: ``/home/mila/flash.log``
as well as the status of the transfer of your home directory to its new location
(``pending``, ``progress`` or ``done``).


How to follow the progress of the transfer ?
---------------------------------------------

Once your transfer is started, the output will be written to a file inside
your current home directory named ``.flash_progress``. Just ``tail ~/.flash_progress``
to get the transfer status.


How to accelerate the transfer ?
---------------------------------

Your whole home directory will be transferred except hidden files, ``.conda/pkgs``,
``.cache`` and other not useful directories.
By default, we use this `list <https://github.com/rubo77/rsync-homedir-excludes/blob/master/rsync-homedir-excludes.txt>`_
of excluded files for the transfer:


You can accelerate the transfer and help
the transition by cleaning-up your home a bit, for example by deleting any unused environments
or even saving them to recreate them later.

Here is an example of how to save your Conda environment


.. prompt:: bash $

    # Activate
    conda activate myenv
    # Export
    conda env export > ~/my_environment.yml

    # Later on the new machine
    conda env create -f ~/my_environment.yml


What happens at the end of the transfer ?
--------------------------------------------

Until the end of the transfer, every time you log into the cluster, you will
use your current (old) home directory. As soon as the transfer finishes, your home
directory will be set to its new location:


.. code-block:: bash

    /home/mila/first_letter/username


**It can take time before the setting is applied** since all machines have a cache
of your account settings so it can take multiple connections before your new home is
effective. If it's an issue, please contact IT support.


Some files are missing ?
-------------------------

Your old home directory will be preserved for some months and will still be available
at its current location ``/network/home/username``. You can manually copy some files from it
until it is deleted.



And Kerberos ?
---------------

The home server does not need a Kerberos ticket to let you access your files as it is secured
with a different mechanism. It means password free ssh access with a private key should work as soon as
you switch to the new one **BUT** some shared file-systems still require Kerberos like ``tmp1`` and
``sec``.

During the transition, all users will still be required to get a Kerberos ticket to start
jobs, running a SLURM command to start a job ``sbatch/salloc`` will guide you in the creation of a keytab
(to automatically renew your ticket with password) and an initial ticket.

If you have any issue with the integrated script, try running

.. prompt:: bash $

    sticket

or contact helpdesk.
