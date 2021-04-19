Docker/Shifter
================


Docker containers are now available on the local cluster with a root-less
system called Shifter integrated into Slurm.
*It is still in beta and be careful with this usage*

Containers
-----------

To first use a container, you have to pull it to the local registry to be
converted to a Shifter-compatible image.

.. prompt:: bash

    shifterimg pull docker:image_name:latest


You can list available images with

.. prompt:: bash

    shifterimg images


**DO NOT USE IMAGES WITH SENSITIVE INFORMATION** yet, it will soon be possible. For now, every image
is pulled to a common registry but access-control will soon be implemented.


Using in Slurm
--------------

Batch job
^^^^^^^^^^

You must use the ``--image=docker:image_name:latest`` directive to specify
the container to use. Once the container is mounted, you are not yet
inside the container's file-system, you must use the ``shifter`` command
to execute a command in the chroot environment of the container.

e.g.:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --image=docker:image_name:latest
    #SBATCH --nodes=1
    #SBATCH --partition=low

    shifter python myPythonScript.py args



Interactive job
^^^^^^^^^^^^^^^^

Using the salloc command, you can request the image while getting the allocation

.. prompt:: bash

    salloc -c2 --mem=16g --image=docker:image_name:latest


and once in the job, you can activate the container's environment with the ``shifter`` command

.. prompt:: bash

    shifter /bin/bash




Command line
-------------

``shifter`` support various options on the command line but you should be
set with the image name and the command to execute:

.. code-block:: bash

     shifter [-h|--help] [-v|--verbose] [--image=<imageType>:<imageTag>]
         [--entrypoint[=command]] [--workdir[=/path]]
         [-E|--clearenv] [-e|--env=<var>=<value>] [--env-file=/env/file
         [-V|--volume=/path/to/bind:/mnt/in/image[:<flags>[,...]][;...]]
         [-m|--module=<modulename>[,...]]
         [-- /command/to/exec/in/shifter [args...]]



Volumes
--------

``/home/yourusername``, ``/Tmp``, ``/ai`` and all ``/network/..`` sub-folders are
mounted inside the container.


GPU
----

To access the GPU inside a container, you need to specify ``--module=nvidia`` on
the ``sbatch/salloc/shifter`` command line

.. prompt:: bash

    shifter --image=centos:7 --module=nvidia bash



Following folders will be mounted in the container:

==============================  ==================  ======================================================
  Host                           Container             Comment
==============================  ==================  ======================================================
/ai/apps/cuda/10.0                /cuda               Cuda libraries and bin, added to ``PATH``
/usr/bin                          /nvidia/bin         To access ``nvidia-smi``
/usr/lib/x86_64-linux-gnu/        /nvidia/lib         ``LD_LIBRARY_PATH`` will be set to ``/nvidia/lib``
==============================  ==================  ======================================================



Remember
---------

- Use image names in 3 parts to avoid confusion: ``_type:name:tag_``
- Please keep in mind that root is squashed on Shifter images, so the software should be installed in a way that is executable to someone with user-level permissions.
- Currently the ``/etc`` and ``/var`` directories are reserved for use by the system and will be overwritten when the image is mounted
- The container is not isolated so you share the network card and all hardware from the host, no need to forward ports


Example
--------

.. code-block:: bash

    username@login-2:~$ shifterimg pull docker:alpine:latest
    2019-10-11T20:12:42 Pulling Image: docker:alpine:latest, status: READY

    username@login-2:~$ salloc -c2 --gres=gpu:1 --image=docker:alpine:latest
    salloc: Granted job allocation 213064
    salloc: Waiting for resource configuration
    salloc: Nodes eos20 are ready for job

    username@eos20:~$ cat /etc/os-release
    NAME="Ubuntu"
    VERSION="18.04.2 LTS (Bionic Beaver)"
    ID=ubuntu
    ID_LIKE=debian
    PRETTY_NAME="Ubuntu 18.04.2 LTS"
    VERSION_ID="18.04"
    VERSION_CODENAME=bionic
    UBUNTU_CODENAME=bionic

    username@eos20:~$ shifter sh
    ~ $ cat /etc/os-release
    NAME="Alpine Linux"
    ID=alpine
    VERSION_ID=3.10.2
    PRETTY_NAME="Alpine Linux v3.10"

    ~ $


.. note::
    Complete Documentation:
    https://docs.nersc.gov/programming/shifter/how-to-use/
