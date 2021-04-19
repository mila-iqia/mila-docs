.. _milacloud:

Mila Cloud Cluster
==================


.. toctree::
   :caption: Particular features
   :maxdepth: 3

   environment
   slurm


Good news, Mila in discussion with cloud providers has received cloud credits for Microsoft Azure and
Google Cloud Compute (GCP) right now.

.. topic:: Do I need to learn how to use those cloud cluster?

   No.
   In the actual phase it is almost the same way you use the Mila-cluster. You will use SLURM, but you will have to connect to a particular login node. In a future phase all will be totally transparent, you will use the same login nodes, and all the data aspect will be taken care of.


Login nodes
-------------
For convenience, a login node has currently been deployed in
*Canada Central Region* (on Azure) and one in the *Central US Region* (GCP).

When the current setup will be stable, the plan is to have one login node per region and per provider.

============    =====================   =====================
  Provider             Region                  IP
============    =====================   =====================
Azure             *not available*
AWS               US                     3.91.250.238
GCP               US                     104.154.137.18
============    =====================   =====================

.. Canada Central+US      13.71.165.214

All login nodes run *Ubuntu 18.04* and support *passwordless* SSH only.


Limits
-------

Current Allocation
^^^^^^^^^^^^^^^^^^

* Azure:
    * 32 x v100 with 16Gb of memory - 1/2/4 per node
    * 66 x k80 with 8Gb of memory (US Region) - 1/2/4 per node
    * 400 cores of cpu only machines: Compute(32Gb) / General(64Gb) / Memory(128Gb)
* GCP:
    * 32 x v100 with 16Gb of memory - 1/2/4 per node
    * 66 x k80 with 8Gb of memory (US Region) - 1/2/4 per node
    * 400 cores of cpu only machines: Compute(32Gb) / General(64Gb) / Memory(128Gb)
* AWS: to come

Instance info:

* Azure: https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu
* AWS: https://aws.amazon.com/ec2/instance-types/
* GCP: https://cloud.google.com/compute/docs/machine-types

Current User Quota
^^^^^^^^^^^^^^^^^^^

+--------+-----------+
| Type   |  Mins     |
+========+===========+
| v100   |  40,000   |
+--------+-----------+
| k80    |  160,000  |
+--------+-----------+
| CPU    |  200,000  |
+--------+-----------+

*Depending on your answers in the form, you can access non-preemptible machines with a 1/4 of the quota.*

To verify your quota on the current cloud against your allocated limits.

.. prompt:: bash $

    squotas



How do I onboard on the cloud?
------------------------------
As the resources are limited, priority will be given according to the nearest deadline and needs.
For those interested in beta testing or using this opportunity for the near deadline, please fill out this form
https://docs.google.com/forms/d/1Q68cAeLUcxk3TGgtFItN_EgGY-EXaBledetJFGWfryY


.. note::
   Contact/info:

   Contact Quentin Lux on Slack and join the `Mila-Cloud-Cluster channel <https://mila-umontreal.slack.com/messages/CG695R7PD>`_
