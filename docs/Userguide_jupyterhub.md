### JupyterHub

**JupyterHub** is a platform connected to SLURM to start a **JupyterLab**
session as a batch job then connects it when the allocation has been granted.
It does not require any ssh tunnel or port redirection, the hub acts as a proxy
server that will redirect you to a session as soon as it is available.

It is currently available for Mila clusters and some Digital Research Alliance
of Canada (Alliance) clusters.

============== ============================================= ============
Cluster        Address                                       Login type
============== ============================================= ============
Mila Local     https://jupyterhub.server.mila.quebec         Google Oauth
Alliance       https://docs.alliancecan.ca/wiki/JupyterHub   DRAC login
============== ============================================= ============

> **WARNING**
> running the session and the SLURM job it is linked to.
>
> To close it, use the ``hub`` menu and then ``Control Panel > Stop my server``

> **NOTE**
> *mila.quebec* account credentials should be used to login and start a
> **JupyterLab** session.

#### Access Mila Storage in JupyterLab

Unfortunately, JupyterLab does not allow the navigation to parent directories of
`$HOME`. This makes some file systems like `/network/datasets` or
`$SLURM_TMPDIR` unavailable through their absolute path in the interface. It
is however possible to create symbolic links to those resources. To do so, you
can use the `ln -s` command:

.. code-block:: bash

   ln -s /network/datasets $HOME

Note that `$SLURM_TMPDIR` is a directory that is dynamically created for each
job so you would need to recreate the symbolic link every time you start a
JupyterHub session:

.. code-block:: bash

   ln -sf $SLURM_TMPDIR $HOME