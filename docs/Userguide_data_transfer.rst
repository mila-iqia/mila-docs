.. _globus_connect_personal:

Data Transmission using Globus Connect Personal
===============================================

Mila doesn't own a Globus license but if the source or destination provides a
Globus account, like Digital Research Alliance of Canada for example, it's
possible to setup Globus Connect Personal to create a personal endpoint on the
Mila cluster and then transfer data to and from the Mila cluster.

Some utilities have been made available to help setup a Globus personal
endpoint:


Login and add a Globus personal endpoint
----------------------------------------

.. code-block:: sh

   # login to your globus account
   /network/datasets/scripts/globus_utils.sh globus whoami

   # add a globus personal endpoint
   /network/datasets/scripts/globus_utils.sh add_endpoint --name "mila-cluster"


Start a Globus personal endpoint
--------------------------------

.. code-block:: sh

   /network/datasets/scripts/globus_utils.sh start_endpoint --dir PATH/TO/DATA

.. note::
   The endpoint used will be the one precedently added. The utility does not
   allow multiple endpoints per user

.. note::
   It's best to use a cpu allocation to handle the Globus personal endpoint.
   Once the personal endpoint is setup, execute the following command to run in
   slurm:

    .. code-block:: sh

       sbatch --ntasks=1 --cpus-per-task=4 --mem=8G /network/datasets/scripts/globus_utils.sh start_endpoint --name "mila-cluster"


Do more with Globus Connect Personal
------------------------------------

To understand better how Globus Personal works and do more with Globus, follow
the Globus guide to `Install, Configure, and Uninstall Globus Connect Personal
for Linux <https://docs.globus.org/how-to/globus-connect-personal-linux/>`_.
