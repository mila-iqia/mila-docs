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


+-----------------------------+--------------------+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
|          Name               |     GPU            | CPUs | Sockets | Cores/Socket | Threads/Core | Memory (GB) | TmpDisk (TB) |  Arch  |   Slurm Features        |
|                             +----------+-----+---+      |         |              |              |             |              |        +-------------------------+
|                             |   Model  | Mem | # |      |         |              |              |             |              |        | GPU Arch and Memory     |
+=============================+==========+=====+===+======+=========+==============+==============+=============+==============+========+=========================+
| :h:`<h5 style="margin: 5px 0 0 0;">GPU Compute Nodes</h5>`                                                                                                      |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-a[001-011]**           | RTX8000  |  48 | 8 |  40  |    2    |      20      |       1      |     384     |      3.6     | x86_64 |      turing,48gb        |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-b[001-005]**           | V100     |  32 | 8 |  40  |    2    |      20      |       1      |     384     |      3.6     | x86_64 |  volta,nvlink,32gb      |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-c[001-040]**           | RTX8000  |  48 | 8 |  64  |    2    |      32      |       1      |     384     |      3       | x86_64 |     turing,48gb         |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-g[001-029]**           | A100     |  80 | 4 |  64  |    2    |      32      |       1      |    1024     |      7       | x86_64 | ampere,nvlink,80gb      |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-i001**                 | A100     |  80 | 4 |  64  |    2    |      32      |       1      |    1024     |      3.6     | x86_64 |     ampere,80gb         |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-j001**                 | A6000    |  48 | 8 |  64  |    2    |      32      |       1      |    1024     |      3.6     | x86_64 |     ampere,48gb         |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-k[001-004]**           | A100     |  40 | 4 |  48  |    2    |      24      |       1      |     512     |      3.6     | x86_64 | ampere,nvlink,40gb      |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">DGX Systems</h5>`                                                                                                            |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-d[001-002]**           | A100     |  40 | 8 |  128 |    2    |      64      |       1      |    1024     |     14       | x86_64 | ampere,nvlink,dgx,40gb  |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-d[003-004]**           | A100     |  80 | 8 |  128 |    2    |      64      |       1      |    2048     |     28       | x86_64 | ampere,nvlink,dgx,80gb  |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-e[002-003]**           | V100     |  32 | 8 |  40  |    2    |      20      |       1      |     512     |      7       | x86_64 |  volta,nvlink,dgx,32gb  |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">CPU Compute Nodes</h5>`                                                                                                      |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-f[001-004]**           | -        |  -  | - |  32  |    1    |      32      |       1      |     256     |     10       | x86_64 |        rome             |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+
| **cn-h[001-004]**           | -        |  -  | - |  64  |    2    |      32      |       1      |     768     |      7       | x86_64 |        milan            |
+-----------------------------+----------+-----+---+------+---------+--------------+--------------+-------------+--------------+--------+-------------------------+


Special nodes and outliers
--------------------------

DGX A100
^^^^^^^^

.. _dgx_a100_nodes:

DGX A100 nodes are NVIDIA appliances with 8 NVIDIA A100 Tensor Core GPUs. Each
GPU has either 40 GB or 80 GB of memory, for a total of 320 GB or 640 GB per
appliance. The GPUs are interconnected via 6 NVSwitches which allow for 600
GB/s point-to-point bandwidth (unidirectional) and a full bisection bandwidth
of 4.8 TB/s (bidirectional). See the table above for the specifications of each
appliance.

In order to run jobs on a DGX A100 with 40GB GPUs, add the flags below to your
Slurm commands::

    --gres=gpu:a100:<number> --constraint="dgx&ampere"

In order to run jobs on a DGX A100 with 80GB GPUs, add the flags below to your
Slurm commands::

    --gres=gpu:a100l:<number> --constraint="dgx&ampere"

MIG
^^^

.. _mig_nodes:

MIG (`Multi-Instance GPU <https://www.nvidia.com/en-us/technologies/multi-instance-gpu/>`_)
is an NVIDIA technology allowing certain GPUs to be
partitioned into multiple *instances*, each of which has a roughly proportional
amount of compute resources, device memory and bandwidth to that memory.

NVIDIA supports MIG on its A100 GPUs and allows slicing the A100 into up to 7
instances. Although this can theoretically be done dynamically, the SLURM job
scheduler does not support doing so in practice as it does not model
reconfigurable resources very well. Therefore, the A100s must currently be
statically partitioned into the required number of instances of every size
expected to be used.

The ``cn-g`` series of nodes include A100-80GB GPUs. A subset have been
configured to offer regular (non-MIG mode) ``a100l`` GPUs. The others have been
configured in MIG mode, and offer the following profiles:

+-----------------------------+----------------------------------------+--------------+
|          Name               |     GPU                                | Cluster-wide |
|                             +----------+---------------+-------------+--------------+
|                             |   Model  |     Memory    |   Compute   |      #       |
+=============================+==========+===============+=============+==============+
| ``a100l.2g.20gb``:h:`<p>`   |          | 20GB :h:`<p>` | 2/7th       |     48       |
| ``a100l.2``                 | A100     | (2/8th)       | *of full*   |              |
+-----------------------------+----------+---------------+-------------+--------------+
| ``a100l.3g.40gb``:h:`<p>`   |          | 40GB :h:`<p>` | 3/7th       |     48       |
| ``a100l.3``                 | A100     | (4/8th)       | *of full*   |              |
+-----------------------------+----------+---------------+-------------+--------------+
| ``a100l.4g.40gb``:h:`<p>`   |          | 40GB :h:`<p>` | 4/7th       |     24       |
| ``a100l.4``                 | A100     | (4/8th)       | *of full*   |              |
+-----------------------------+----------+---------------+-------------+--------------+

And can be requested using a SLURM flag such as ``--gres=gpu:a100l.2``

The partitioning may be revised as needs and SLURM capabilities evolve. Other
MIG profiles exist and could be introduced.


.. warning::

    MIG has a number of `important limitations <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#app-considerations>`_,
    most notably that a GPU in MIG mode does not support graphics APIs
    (OpenGL/Vulkan), nor P2P over NVLink and PCIe. We have therefore chosen to
    limit every MIG job to exactly one MIG slice and no more. Thus,
    ``--gres=gpu:a100l.3`` will work (*and request a size-3 slice of an*
    ``a100l`` *GPU*) but ``--gres=gpu:a100l.2:3`` (*with* ``:3`` *requesting
    three size-1 slices*) **will not**.
