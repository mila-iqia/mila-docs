.. _Using containers:

Using containers
================

Docker containers are now available on the local cluster without root priviledges using `podman <https://podman.io>`_.

Generally any command-line argument accepted by docker will work with podman. this means that you can mostly use docker examples in you find on the web by replacing docker with podman in the command line.

.. note::
    Complete Podman Documentation:
    https://docs.podman.io/en/stable/

Using in SLURM
--------------

To use podman you can just use the podman command in either a batch script or an interactive job.

One difference in configuration is that for certain technical reasons all the storage for podman (images, containers, ...) is on a job-specific location and will be lost after the job is complete or preempted. If you have data that must be preseved across jobs, you can `mount <https://docs.podman.io/en/v5.2.4/markdown/podman-run.1.html#mount-type-type-type-specific-option>` a local folder inside the container, such as $SCRATCH or you home to save data.

.. code-block:: bash

   $ podman run --mount type=bind,source=$SCRATCH/exp,destination=/data/exp bash touch /data/exp/file
   $ ls $SCRATCH/exp
   file

You can use multiple containers in a single job, but you have to be careful about the memory and CPU limits of the job.

GPU
---

To use a GPU you need to a GPU job and then use the `--device nvidia.com/gpu=all` for all GPUs allocated to the job or `--device nvidia.com/gpu=n` where n is the gpu you want in the container, starting at 0.


.. code-block:: bash

   $ nvidia-smi
   $ podman run --device nvidia.com/gpu=all nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
   $ podman run --device nvidia.com/gpu=0 nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
   $ podman run --device nvidia.com/gpu=1 nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi

You can pass `--device` multiple times to add more than one gpus to the container.

.. note::
   CDI (GPU) support documentation:
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html#running-a-workload-with-cdi
