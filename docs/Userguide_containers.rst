.. _Using containers:

Using containers
================

Docker containers are now available on the local cluster without root
priviledges using `podman <https://podman.io>`_.

Generally any command-line argument accepted by docker will work with podman.
This means that you can mostly use the docker examples you find on the web by
replacing `docker` with `podman` in the command line.

.. note::
    Complete Podman Documentation: https://docs.podman.io/en/stable/

Using in SLURM
--------------

To use podman you can just use the `podman` command in either a batch script or
an interactive job.

One difference in configuration is that for certain technical reasons all the
storage for podman (images, containers, ...) is on a job-specific location and
will be lost after the job is complete or preempted. If you have data that must
be preseved across jobs, you can `mount
<https://docs.podman.io/en/v5.2.4/markdown/podman-run.1.html#mount-type-type-type-specific-option>`_
a local folder inside the container, such as `$SCRATCH` or your home to save
data.

.. code-block:: bash

   $ podman run --mount type=bind,source=$SCRATCH/exp,destination=/data/exp bash touch /data/exp/file
   $ ls $SCRATCH/exp
   file

You can use multiple containers in a single job, but you have to be careful
about the memory and CPU limits of the job.

.. note::

   Due to the cluster environment you may see warning messages like
   `WARN[0000] "/" is not a shared mount, this could cause issues or missing mounts with rootless containers`,
   `ERRO[0000] cannot find UID/GID for user <user>: no subuid ranges found for user "<user>" in /etc/subuid - check rootless mode in man pages.`,
   `WARN[0000] Using rootless single mapping into the namespace. This might break some images. Check /etc/subuid and /etc/subgid for adding sub*ids if not using a network user`
   or
   `WARN[0005] Failed to add pause process to systemd sandbox cgroup: dbus: couldn't determine address of session bus`
   but as far as we can see those can be safely ignored and should not have
   an impact on your images.

GPU
---

To use a GPU in a container, you need to a GPU job and then use ``--device
nvidia.com/gpu=all`` to make all GPUs allocated available in the container or
``--device nvidia.com/gpu=N`` where `N` is the gpu index you want in the
container, starting at 0.


.. code-block:: bash

  $ nvidia-smi
  Fri Dec 13 12:47:34 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA L40S                    On  |   00000000:4A:00.0 Off |                    0 |
  | N/A   25C    P8             36W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+
  |   1  NVIDIA L40S                    On  |   00000000:61:00.0 Off |                    0 |
  | N/A   26C    P8             35W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+
  $ podman run --device nvidia.com/gpu=all nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
  Fri Dec 13 17:48:21 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA L40S                    On  |   00000000:4A:00.0 Off |                    0 |
  | N/A   25C    P8             36W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+
  |   1  NVIDIA L40S                    On  |   00000000:61:00.0 Off |                    0 |
  | N/A   25C    P8             35W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+
  $ podman run --device nvidia.com/gpu=0 nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
  Fri Dec 13 17:48:33 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA L40S                    On  |   00000000:4A:00.0 Off |                    0 |
  | N/A   25C    P8             36W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+
  $ podman run --device nvidia.com/gpu=1 nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
  Fri Dec 13 17:48:40 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA L40S                    On  |   00000000:61:00.0 Off |                    0 |
  | N/A   25C    P8             35W /  350W |       1MiB /  46068MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+

You can pass ``--device`` multiple times to add more than one gpus to the container.

.. note::
   CDI (GPU) support documentation:
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html#running-a-workload-with-cdi
