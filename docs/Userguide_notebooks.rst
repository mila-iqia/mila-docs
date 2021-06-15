Tools
======


JupyterHub
-----------

**JupyterHub** is a platform connected to SLURM to start Jupyter notebook as
a batch job in order to connect when the allocation has been granted. It does not require any ssh tunnel or port redirection,
the hub acts as a proxy server that will redirect you to your notebook as soon as it is available.

It is currently available for Mila clusters and Compute Canada Helios cluster (under PBS)

========================== ==================================================================== ================
Cluster                     Address                                                              Login type
========================== ==================================================================== ================
Mila Local                  https://jupyterhub.server.mila.quebec                                 Google Oauth
Mila Cloud GCP              https://jupyterhub.gcp.mila.quebec                                    Google Oauth
CC Helios                   https://jupyter.calculquebec.ca/                                      CC login
========================== ==================================================================== ================


.. warning::
    Do not forget to close your notebook/lab! Closing the window leaves the notebook running.

    To close it, use the ``hub`` menu and then ``Control Panel > Stop my server``


.. note::

  **For Mila Clusters:**

   | Currently you can use your Google *mila.quebec* account to login and start a **JupyterLab**.
   | The authentication process matches your *mila.quebec* login to your cluster id:
   |
   | ``my_username@mila.quebec`` => ``my_username``
   |
   | If the two are different, contact the Mila sysadmins specifying your correct credentials

  **For Mila Cloud Clusters:**

   | Be sure that you have requested and been allocated access to it. See `How do I onboard the Cloud? <https://docs.mila.quebec/mila-cloud-cluster/index.html#how-do-i-onboard-on-the-cloud>`_
