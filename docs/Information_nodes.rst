Node profile description
========================

.. _node_list:


.. role:: h(raw)
   :format: html

..
   Je trouve cela un peu futile de maintenir cette documentation à jour
   manuellement.  Peut-être pourrions nous créer dans ce dossier des sripts qui
   pourraient créer une entrée RST et qui pourraient être exécutés sur un noeud
   au Mila pour les mises à jour.


+---------------------------------------+--------------+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
|               Name                    |     GPU      | CPUs | Sockets | Cores/Socket | Threads/Core | Memory (GB) | TmpDisk (TB) |  Arch  |   Slurm Features    |
|                                       +----------+---+      |         |              |              |             |              |        +---------------------+
|                                       |   Model  | # |      |         |              |              |             |              |        | GPU Arch and Memory |
+=======================================+==========+===+======+=========+==============+==============+=============+==============+========+=====================+
| :h:`<h5 style="margin: 5px 0 0 0;">GPU Compute Nodes</h5>`                                                                                                      |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| **cn-a[001-011]**                     | rtx8000  | 8 |  80  |    2    |      20      |       2      |     384     |      3.6     | x86_64 |      turing,48gb    |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| **cn-b[001-005]**                     | V100     | 8 |  80  |    2    |      20      |       2      |     384     |      3.6     | x86_64 |  volta,nvlink,32gb  |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| **cn-c[001-040]**                     | rtx8000  | 8 |  64  |    2    |      32      |       1      |     384     |      3       | x86_64 |     turing,48gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">DGX Systems</h5>`                                                                                                            |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| mila01                                | V100     | 8 |  80  |    2    |      20      |       2      |     512     |      7       | x86_64 |      volta,16gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| mila[02-03]                           | V100     | 8 |  80  |    2    |      20      |       2      |     512     |      7       | x86_64 |      volta,32gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| **cn-d[001-002]**                     | A100     | 8 |  128 |    2    |      64      |       2      |    1024     |     14       | x86_64 | ampere,nvlink,40gb  |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">Legacy GPU</h5>`                                                                                                             |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler3                               | k80      | 8 |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |     kepler,12gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler4                               | m40      | 4 |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |    maxwell,24gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler5                               | V100     | 2 |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |      volta,16gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">TITAN RTX</h5>`                                                                                                              |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| rtx[6,9]                              | titanrtx | 2 |  20  |    1    |      10      |       2      |     128     |      3.6     | x86_64 |     turing,24gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| rtx[1-5,7-8]                          | titanrtx | 2 |  20  |    1    |      10      |       2      |     128     |      0.93    | x86_64 |     turing,24gb     |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">POWER9</h5>`                                                                                                                 |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| power9[1-2]                           | v100     | 4 |  128 |    2    |      16      |       4      |     586     |      0.88    | power9 |  volta,nvlink,16gb  |
+---------------------------------------+----------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+


Special nodes and outliers
--------------------------

DGX A100
^^^^^^^^

.. _dgx_a100_nodes:

DGX A100 nodes are NVIDIA appliances with 8 NVIDIA A100 Tensor Core GPUs. Each
GPU has 40 GB of memory, for a total of 320 GB per appliance. The GPUs are
interconnected via 6 NVSwitches which allows 4.8 TB/s bi-directional bandwidth.

In order to run jobs on a DGX A100, add the flags below to your Slurm
commands::

    --gres=gpu:a100:<number> --reservation=DGXA100

Power9
^^^^^^

.. _power9_nodes:

Power9_ nodes are using a different processor instruction set than Intel and
AMD (x86_64) based nodes. As such you need to setup your environment again
for those nodes specifically.

* Power9 nodes have 128 threads. (2 processors / 16 cores / 4 way SMT)
* 4 x V100 SMX2 (16 GB) with NVLink
* In a Power9 node GPUs and CPUs communicate with each other using NVLink
  instead of PCIe. This allow them to communicate quickly between each other.
  More on Large Model Support (LMS_)

Power9 nodes have the same software stack as the regular nodes and each
software should be included to deploy your environment as on a regular node.


.. _LMS: https://developer.ibm.com/articles/performance-results-with-lmstf2/
.. _Power9: https://en.wikipedia.org/wiki/POWER9

.. .. prompt:: bash $, auto
..
..     # on Mila cluster's login node
..     $ srun -c 1 --reservation=power9 --pty bash
..
..     # setup anaconda
..     $ wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-ppc64le.sh
..     $ chmod +x Anaconda3-2019.07-Linux-ppc64le.sh
..     $ module load anaconda/3
..
..     $ conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
..     $ conda create -n p9 python=3.6
..     $ conda activate p9
..     $ conda install powerai=1.6.0
..
..     # setup is done!


AMD
^^^

.. warning::

    As of August 20 2019 the GPUs had to return back to AMD.  Mila will get
    more samples. You can join the amd_ slack channels to get the latest
    information

.. _amd: https://mila-umontreal.slack.com/archives/CKV5YKEP6/p1561471261000500

Mila has a few node equipped with MI50_ GPUs.

.. _MI50: https://www.amd.com/en/products/professional-graphics/instinct-mi50

.. prompt:: bash $, auto

    $ srun --gres=gpu -c 8 --reservation=AMD --pty bash

    # first time setup of AMD stack
    $ conda create -n rocm python=3.6
    $ conda activate rocm

    $ pip install tensorflow-rocm
    $ pip install /wheels/pytorch/torch-1.1.0a0+d8b9d32-cp36-cp36m-linux_x86_64.whl
