Building the containers
-----------------------

Building a container is like creating a new environment except that containers
are much more powerful since they are self-contained systems.  With
singularity, there are two ways to build containers.

The first one is by yourself, it's like when you got a new Linux laptop and you
don't really know what you need, if you see that something is missing, you
install it. Here you can get a vanilla container with Ubuntu called a sandbox,
you log in and you install each packages by yourself.  This procedure can take
time but will allow you to understand how things work and what you need. This is
recommended if you need to figure out how things will be compiled or if you want
to install packages on the fly. We'll refer to this procedure as singularity
sandboxes.

The second way is more like you know what you want, so you write a list of
everything you need, you send it to singularity and it will install everything
for you. Those lists are called singularity recipes.


First way: Build and use a sandbox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might ask yourself: *On which machine should I build a container?*

First of all, you need to choose where you'll build your container. This
operation requires **memory and high cpu usage**.

.. warning:: Do NOT build containers on any login nodes !

* (Recommended for beginner) If you need to **use apt-get**, you should **build
  the container on your laptop** with sudo privileges. You'll only need to
  install singularity on your laptop. Windows/Mac users can look `there`_ and
  Ubuntu/Debian users can use directly:

        .. _there: https://www.sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac

        .. prompt:: bash $

                sudo apt-get install singularity-container


* If you **can't install singularity** on your laptop and you **don't need
  apt-get**, you can reserve a **cpu node on the Mila cluster** to build your
  container.


In this case, in order to avoid too much I/O over the network, you should define
the singularity cache locally:

        .. prompt:: bash $

                export SINGULARITY_CACHEDIR=$SLURM_TMPDIR

* If you **can't install singularity** on your laptop and you **want to use
  apt-get**, you can use `singularity-hub`_ to build your containers and read
  Recipe_section_.

.. _singularity-hub: https://www.singularity-hub.org/


Download containers from the web
""""""""""""""""""""""""""""""""

Hopefully, you may not need to create containers from scratch as many have been
already built for the most common deep learning software. You can find most of
them on `dockerhub`_.

.. _dockerhub: https://hub.docker.com/

Go on `dockerhub`_ and select the container you want to pull.

.. _dockerhub: https://hub.docker.com/

For example, if you want to get the latest PyTorch version with GPU support
(Replace *runtime* by *devel* if you need the full Cuda toolkit):

.. prompt:: bash $

        singularity pull docker://pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

Or the latest TensorFlow:

.. prompt:: bash $

        singularity pull docker://tensorflow/tensorflow:latest-gpu-py3

Currently the pulled image ``pytorch.simg`` or ``tensorflow.simg`` is read-only
meaning that you won't be able to install anything on it.  Starting now, PyTorch
will be taken as example. If you use TensorFlow, simply replace every
**pytorch** occurrences by **tensorflow**.

How to add or install stuff in a container
""""""""""""""""""""""""""""""""""""""""""

The first step is to transform your read only container
``pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg`` in a writable version that will
allow you to add packages.

.. warning:: Depending on the version of singularity you are using, singularity
   will build a container with the extension .simg or .sif. If you're using
   .sif files, replace every occurences of .simg by .sif.

.. tip:: If you want to use **apt-get** you have to put **sudo** ahead of the
   following commands

This command will create a writable image in the folder ``pytorch``.

.. prompt:: bash $

   singularity build --sandbox pytorch pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg


Then you'll need the following command to log inside the container.

.. prompt:: bash $

   singularity shell --writable -H $HOME:/home pytorch


Once you get into the container, you can use pip and install anything you need
(Or with ``apt-get`` if you built the container with sudo).

.. warning:: Singularity mounts your home folder, so if you install things into
   the ``$HOME`` of your container, they will be installed in your real
   ``$HOME``!


You should install your stuff in /usr/local instead.


Creating useful directories
"""""""""""""""""""""""""""

One of the benefits of containers is that you'll be able to use them across
different clusters. However for each cluster the dataset and experiment folder
location can be different. In order to be invariant to those locations, we will
create some useful mount points inside the container:

.. prompt:: bash <Singularity_container>$

   mkdir /dataset
   mkdir /tmp_log
   mkdir /final_log


From now, you won't need to worry anymore when you write your code to specify
where to pick up your dataset. Your dataset will always be in ``/dataset``
independently of the cluster you are using.


Testing
"""""""

If you have some code that you want to test before finalizing your container,
you have two choices.  You can either log into your container and run Python
code inside it with:

.. prompt:: bash $

        singularity shell --nv pytorch

Or you can execute your command directly with

.. prompt:: bash $

   singularity exec --nv pytorch Python YOUR_CODE.py

.. tip:: ---nv allows the container to use gpus. You don't need this if you
   don't plan to use a gpu.

.. warning:: Don't forget to clear the cache of the packages you installed in
   the containers.


Creating a new image from the sandbox
"""""""""""""""""""""""""""""""""""""

Once everything you need is installed inside the container, you need to convert
it back to a read-only singularity image with:

.. prompt:: bash $

   singularity build pytorch_final.simg pytorch

.. _Recipe_section:

Second way: Use recipes
^^^^^^^^^^^^^^^^^^^^^^^

A singularity recipe is a file including specifics about installation software,
environment variables, files to add, and container metadata.  It is a starting
point for designing any custom container. Instead of pulling a container and
installing your packages manually, you can specify in this file the packages
you want and then build your container from this file.

Here is a toy example of a singularity recipe installing some stuff:

.. code-block:: bash

   ################# Header: Define the base system you want to use ################
   # Reference of the kind of base you want to use (e.g., docker, debootstrap, shub).
   Bootstrap: docker
   # Select the docker image you want to use (Here we choose tensorflow)
   From: tensorflow/tensorflow:latest-gpu-py3

   ################# Section: Defining the system #################################
   # Commands in the %post section are executed within the container.
   %post
           echo "Installing Tools with apt-get"
           apt-get update
           apt-get install -y cmake libcupti-dev libyaml-dev wget unzip
           apt-get clean
           echo "Installing things with pip"
           pip install tqdm
           echo "Creating mount points"
           mkdir /dataset
           mkdir /tmp_log
           mkdir /final_log


   # Environment variables that should be sourced at runtime.
   %environment
           # use bash as default shell
           SHELL=/bin/bash
           export SHELL


A recipe file contains two parts: the ``header`` and ``sections``. In the
``header`` you specify which base system you want to use, it can be any docker
or singularity container. In ``sections``, you can list the things you want to
install in the subsection ``post`` or list the environment's variable you need
to source at each runtime in the subsection ``environment``. For a more detailed
description, please look at the `singularity documentation`_.

.. _singularity documentation: https://www.sylabs.io/guides/2.6/user-guide/container_recipes.html#container-recipes

In order to build a singularity container from a singularity recipe file, you
should use:

.. prompt:: bash $

   sudo singularity build <NAME_CONTAINER> <YOUR_RECIPE_FILES>

.. warning:: You always need to use sudo when you build a container from a
   recipe. As there is no access to sudo on the cluster, a personal computer or
   the use singularity hub is needed to build a container


Build recipe on singularity hub
"""""""""""""""""""""""""""""""

Singularity hub allows users to build containers from recipes directly on
singularity-hub's cloud meaning that you don't need to build containers by
yourself.  You need to register on `singularity-hub`_ and link your
singularity-hub account to your GitHub account, then :

.. _singularity-hub: https://www.singularity-hub.org/

   1) Create a new github repository.
   2) Add a collection on `singularity-hub`_ and select the github repository your created.
   3) Clone the github repository on your computer.
   4) Write the singularity recipe and save it as a file nammed **Singularity**.
   5) Git add **Singularity**, commit and push on the master branch.

At this point, robots from singularity-hub will build the container for you, you
will be able to download your container from the website or directly with:

.. prompt:: bash $

        singularity pull shub://<github_username>/<repository_name>


Example: Recipe with OpenAI gym, MuJoCo and Miniworld
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Here is an example on how you can use singularity recipe to install complex
environment as OpenAI gym, MuJoCo and Miniworld on a PyTorch based container.
In order to use MuJoCo, you'll need to copy the key stored on the Mila cluster
in `/ai/apps/mujoco/license/mjkey.txt` to your current directory.

.. code-block:: bash

   #This is a dockerfile that sets up a full Gym install with test dependencies
   Bootstrap: docker

   # Here we ll build our container upon the pytorch container
   From: pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

   # Now we'll copy the mjkey file located in the current directory inside the container's root
   # directory
   %files
           mjkey.txt

   # Then we put everything we need to install
   %post
           export PATH=$PATH:/opt/conda/bin
           apt -y update && \
           apt install -y keyboard-configuration && \
           apt install -y \
           python3-dev \
           python-pyglet \
           python3-opengl \
           libhdf5-dev \
           libjpeg-dev \
           libboost-all-dev \
           libsdl2-dev \
           libosmesa6-dev \
           patchelf \
           ffmpeg \
           xvfb \
           libhdf5-dev \
           openjdk-8-jdk \
           wget \
           git \
           unzip && \
           apt clean && \
           rm -rf /var/lib/apt/lists/*
           pip install h5py

           # Download Gym and MuJoCo
           mkdir /Gym && cd /Gym
           git clone https://github.com/openai/gym.git || true && \
           mkdir /Gym/.mujoco && cd /Gym/.mujoco
           wget https://www.roboti.us/download/mjpro150_linux.zip  && \
           unzip mjpro150_linux.zip && \
           wget https://www.roboti.us/download/mujoco200_linux.zip && \
           unzip mujoco200_linux.zip && \
           mv mujoco200_linux mujoco200

           # Export global environment variables
           export MUJOCO_PY_MJKEY_PATH=/Gym/.mujoco/mjkey.txt
           export MUJOCO_PY_MUJOCO_PATH=/Gym/.mujoco/mujoco150/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mjpro150/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mujoco200/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
           cp /mjkey.txt /Gym/.mujoco/mjkey.txt
           # Install Python dependencies
           wget https://raw.githubusercontent.com/openai/mujoco-py/master/requirements.txt
           pip install -r requirements.txt
           # Install Gym and MuJoCo
           cd /Gym/gym
           pip install -e '.[all]'
           # Change permission to use mujoco_py as non sudoer user
           chmod -R 777 /opt/conda/lib/python3.6/site-packages/mujoco_py/
           pip install --upgrade minerl

   # Export global environment variables
   %environment
           export SHELL=/bin/sh
           export MUJOCO_PY_MJKEY_PATH=/Gym/.mujoco/mjkey.txt
           export MUJOCO_PY_MUJOCO_PATH=/Gym/.mujoco/mujoco150/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mjpro150/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mujoco200/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
           export PATH=/Gym/gym/.tox/py3/bin:$PATH

   %runscript
           exec /bin/sh "$@"


Here is the same recipe but written for TensorFlow:

.. code-block:: bash

   #This is a dockerfile that sets up a full Gym install with test dependencies
   Bootstrap: docker

   # Here we ll build our container upon the tensorflow container
   From: tensorflow/tensorflow:latest-gpu-py3

   # Now we'll copy the mjkey file located in the current directory inside the container's root
   # directory
   %files
           mjkey.txt

   # Then we put everything we need to install
   %post
           apt -y update && \
           apt install -y keyboard-configuration && \
           apt install -y \
           python3-setuptools \
           python3-dev \
           python-pyglet \
           python3-opengl \
           libjpeg-dev \
           libboost-all-dev \
           libsdl2-dev \
           libosmesa6-dev \
           patchelf \
           ffmpeg \
           xvfb \
           wget \
           git \
           unzip && \
           apt clean && \
           rm -rf /var/lib/apt/lists/*

           # Download Gym and MuJoCo
           mkdir /Gym && cd /Gym
           git clone https://github.com/openai/gym.git || true && \
           mkdir /Gym/.mujoco && cd /Gym/.mujoco
           wget https://www.roboti.us/download/mjpro150_linux.zip  && \
           unzip mjpro150_linux.zip && \
           wget https://www.roboti.us/download/mujoco200_linux.zip && \
           unzip mujoco200_linux.zip && \
           mv mujoco200_linux mujoco200

           # Export global environment variables
           export MUJOCO_PY_MJKEY_PATH=/Gym/.mujoco/mjkey.txt
           export MUJOCO_PY_MUJOCO_PATH=/Gym/.mujoco/mujoco150/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mjpro150/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mujoco200/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
           cp /mjkey.txt /Gym/.mujoco/mjkey.txt

           # Install Python dependencies
           wget https://raw.githubusercontent.com/openai/mujoco-py/master/requirements.txt
           pip install -r requirements.txt
           # Install Gym and MuJoCo
           cd /Gym/gym
           pip install -e '.[all]'
           # Change permission to use mujoco_py as non sudoer user
           chmod -R 777 /usr/local/lib/python3.5/dist-packages/mujoco_py/

           # Then install miniworld
           cd /usr/local/
           git clone https://github.com/maximecb/gym-miniworld.git
           cd gym-miniworld
           pip install -e .

   # Export global environment variables
   %environment
           export SHELL=/bin/bash
           export MUJOCO_PY_MJKEY_PATH=/Gym/.mujoco/mjkey.txt
           export MUJOCO_PY_MUJOCO_PATH=/Gym/.mujoco/mujoco150/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mjpro150/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Gym/.mujoco/mujoco200/bin
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
           export PATH=/Gym/gym/.tox/py3/bin:$PATH

   %runscript
           exec /bin/bash "$@"


Keep in mind that those environment variables are sourced at runtime and not at
build time. This is why, you should also define them in the ``%post`` section
since they are required to install MuJoCo.

