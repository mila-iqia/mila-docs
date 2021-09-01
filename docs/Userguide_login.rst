Logging in to the cluster
=========================

To access the Mila Cluster clusters, you will need an account. Please contact
Mila systems administrators if you don't have it already. Our IT support service
is available here: https://it-support.mila.quebec/


SSH Login
---------

You can access the Mila cluster via ssh:

.. prompt:: bash $

    ssh <user>@login.server.mila.quebec -p 2222

Four login nodes are available and accessible behind a load balancer. At each
connection, you will be redirected to the least loaded login-node. Each login
node can be directly accessed via: ``login-X.login.server.mila.quebec`` on port
``2222``.

The login nodes support the following authentication mechanisms:
``publickey,keyboard-interactive``.  If you would like to set an entry in your
``.ssh/config`` file, please use the following recommendation:

.. code-block:: bash

   Host HOSTALIAS
       User YOUR-USERNAME
       Hostname login.server.mila.quebec
       PreferredAuthentications publickey,keyboard-interactive
       Port 2222
       ServerAliveInterval 120
       ServerAliveCountMax 5


The RSA, DSA and ECDSA fingerprints for Mila's login nodes are:

.. code-block:: bash

    SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8 (ECDSA)
    SHA256:XvukABPjV75guEgJX1rNxlDlaEg+IqQzUnPiGJ4VRMM (DSA)
    SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o (RSA)
    SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg (ED25519)

