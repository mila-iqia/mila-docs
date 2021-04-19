Reproducible.. _singularity:

Singularity
############

.. contents:: Mila Cluster
   :depth: 2
   :local:


.. highlight:: bash

Singularity is a software container system designed to facilitate portability and reproducibility of high performance computing (HPC) workflows.
It performs a function similar to docker, but with HPC in mind. It is compatible with existing docker containers, and provides tools for
building new containers from recipe files or ad-hoc commands.

Why should you use containers ?
===============================

================ =========================================================================================
Advantages       Descriptions
================ =========================================================================================
Reproducibility  All software you have used for your experiments is packaged inside one file.
Portability      Copy the container you built on every cluster without the need to install anything.
Flexibility      You can use apt-get !
================ =========================================================================================

Building containers
===================

Building a container is like creating a new environment except that containers are much more powerful since they are self-contain systems.
With singularity, there are two ways to build containers.

The first one is by yourself, it's like when you got a new Linux laptop and you don't really know what you need, if you see that something is missing,
you install it. Here you can get a vanilla container with Ubuntu called a sandbox, you log in and you install each packages by yourself.
This procedure can take time but will allow you to understand how things work and what you need. This is recommended if you need to figure out
how things will be compiled or if you want to install packages on the fly. We'll refer to this procedure as singularity sandboxes.


The second one way is more like you know what you want, so you write a list of everything you need, you sent it to singularity and it will install
everything for you. Those lists are called singularity recipes.


First way: Build and use a sandbox
..................................

On which machine should I build a container ?
-----------------------------------------------

First of all, you need to choose where you'll build your container. This operation requires **memory and high cpu usage**.

.. warning:: Do NOT build containers on any login nodes !

* (Recommended for beginner) If you need to **use apt-get**, you should **build the container on your laptop** with sudo privileges. You'll only need to install singularity on your laptop. Windows/Mac users can look `there`_ and Ubuntu/Debian users can use directly:

        .. _there: https://www.sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac

        .. prompt:: bash $

                sudo apt-get install singularity-container


* If you **can't install singularity** on your laptop and you **don't need apt-get**, you can reserve a **cpu node on the mila cluster** to build your container.


In this case, in order to avoid too much I/O over the network, you should define the singularity cache locally:

        .. prompt:: bash $

                export SINGULARITY_CACHEDIR=$SLURM_TMPDIR

* If you **can't install singularity** on your laptop and you **want to use apt-get**, you can use `singularity-hub`_ to build your containers and read Recipe_section_.

.. _singularity-hub: https://www.singularity-hub.org/


Download containers from the web
--------------------------------

Hopefully, you may not need to create containers from scratch as many have been already built for the most common deep learning software.
You can find most of them on `dockerhub`_.

.. _dockerhub: https://hub.docker.com/


.. tip::
        (Optional) You can also pull containers from nvidia cloud see :ref:`nvidia`

Go on `dockerhub`_ and select the container you want to pull.

.. _dockerhub: https://hub.docker.com/

For example, if you want to get the latest pytorch version with gpu support (Replace *runtime* by *devel* if you need the full CUDA toolkit):

.. prompt:: bash $

        singularity pull docker://pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

or the latest tensorflow:

.. prompt:: bash $

        singularity pull docker://tensorflow/tensorflow:latest-gpu-py3

Currently the pulled image ``pytorch.simg`` or ``tensorflow.simg`` is read only meaning that you won't be able to install anything on it.
Starting now, pytorch will be taken as example. If you use tensorflow, simply replace every **pytorch** occurrences by **tensorflow**.

How to add or install stuff in a container
------------------------------------------

The first step is to transform your read only container ``pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg`` in a writable version that will allow you to add packages.

.. warning:: Depending of the version of singularity you are using, singularity will build a container with the extension .simg or .sif. If you got .sif files, replace every occurences of .simg by .sif.

.. tip::
        If you want to use **apt-get** you have to put **sudo** ahead of the following commands

This command will create a writable image in the folder ``pytorch``.

.. prompt:: bash $

        singularity build --sandbox pytorch pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg


Then you'll need the following command to log inside the container.

.. prompt:: bash $

        singularity shell --writable -H $HOME:/home pytorch


Once you get into the container, you can use pip and install anything you need (Or with ``apt-get`` if you built the container with sudo).

.. warning:: Singularity mount your home, so if you install things into the $HOME of your container, they will be installed in your real $HOME !


You should install your stuff in /usr/local instead.

Creating useful directory
^^^^^^^^^^^^^^^^^^^^^^^^^

One of the benefit of containers is that you'll be able to use them across different clusters. However for each cluster the dataset and experiment folder location
can be different. In order to be invariant to those locations, we will create some useful mount points inside the container:

.. prompt:: bash <Singularity_container>$

        mkdir /dataset
        mkdir /tmp_log
        mkdir /final_log


From now, you won't need to worry anymore when you write your code to specify where to pick up your dataset. Your dataset will always be in ``/dataset``
independently of the cluster you are using.

Testing
^^^^^^^

If you have some code that you want to test before finalizing your container, you have two choices.
You can either log into your container and run python code inside it with

.. prompt:: bash $

        singularity shell --nv pytorch

or you can execute your command directly with

.. prompt:: bash $

        singularity exec --nv pytorch python YOUR_CODE.py

.. tip:: ---nv allows the container to use gpus. You don't need this if you don't plan to use a gpu.

.. warning:: Don't forget to clear the cache of the packages you installed in the containers.

Creating a new image from the sandbox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once everything you need is installed inside the container, you need to convert it back to a read-only singularity image with:

.. prompt:: bash $

        singularity build pytorch_final.simg pytorch

.. _Recipe_section:

Second way: Use recipes
.......................

Singularity Recipe
------------------

A singularity recipe is a file including specifics about installation software, environment variables, files to add, and container metadata.
It is a starting point for designing any custom container. Instead of pulling a container and install your packages manually, you can specify
in this file the packages you want and then build your container from this file.

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


A recipe file contains two parts: the ``header`` and ``sections``. In the ``header`` you specify which base system you want to
use, it can be any docker or singularity container. In ``sections``, you can list the things you want to install in the subsection
``post`` or list the environment's variable you need to source at each runtime in the subsection ``environment``. For a more detailed
description, please look at the `singularity documentation`_.

.. _singularity documentation: https://www.sylabs.io/guides/2.6/user-guide/container_recipes.html#container-recipes

In order to build a singularity container from a singularity recipe file, you should use:

.. prompt:: bash $

        sudo singularity build <NAME_CONTAINER> <YOUR_RECIPE_FILES>

.. warning:: You always need to use sudo when you build a container from a recipe.


Build recipe on singularity hub
-------------------------------

Singularity hub allows users to build containers from recipes directly on singularity-hub's cloud meaning that you don't need anymore to build containers by yourself.
You need to register on `singularity-hub`_ and link your singularity-hub account to your github account, then

.. _singularity-hub: https://www.singularity-hub.org/

        1) Create a new github repository.
        2) Add a collection on `singularity-hub`_ and select the github repository your created.
        3) Clone the github repository on your computer.
        4) Write the singularity recipe and save it as a file nammed **Singularity**.
        5) Git add **Singularity**, commit and push on the master branch.

At this point, robots from singularity-hub will build the container for you, you will be able to download your container from the website or directly with:

.. prompt:: bash $

        singularity pull shub://<github_username>/<repository_name>


Example: Recipe with openai gym, mujoco and miniworld
-----------------------------------------------------

Here is an example on how you can use singularity recipe to install complex environment as opanai gym, mujoco and miniworld on a pytorch based container.
In order to use mujoco, you'll need to copy the key stored on the mila cluster in `/ai/apps/mujoco/license/mjkey.txt` to your current directory.

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

                # Download Gym and Mujoco
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
                # Install python dependencies
                wget https://raw.githubusercontent.com/openai/mujoco-py/master/requirements.txt
                pip install -r requirements.txt
                # Install Gym and Mujoco
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


Here is the same recipe but written for TensorFlow.

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

                # Download Gym and Mujoco
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

                # Install python dependencies
                wget https://raw.githubusercontent.com/openai/mujoco-py/master/requirements.txt
                pip install -r requirements.txt
                # Install Gym and Mujoco
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


Keep in mind that those environment variables are sourced at runtime and not at build time. This is why, you should also define them in the ``%post`` section since they are required to install mujuco.


Using containers on clusters
============================

On every cluster with SLURM, dataset and intermediate results should go in ``$SLURM_TMPDIR`` while the final experiments results should go in ``$SCRATCH``.
In order to use the container you built, you need to copy it on the cluster you want to use.

.. warning:: You should always store your container in $SCRATCH !

Then reserve a node with srun/sbatch, copy the container and your dataset on the node given by slurm (i.e in ``$SLURM_TMPDIR``) and execute the code ``<YOUR_CODE>`` within the container ``<YOUR_CONTAINER>`` with:

.. prompt:: bash $

        singularity exec --nv -H $HOME:/home -B $SLURM_TMPDIR:/dataset/ -B $SLURM_TMPDIR:/tmp_log/ -B $SCRATCH:/final_log/ $SLURM_TMPDIR/<YOUR_CONTAINER> python <YOUR_CODE>


Remember that ``/dataset``, ``/tmp_log`` and ``/final_log`` were created in the previous section. Now each time, we'll use singularity, we are
explicitly telling it to mount ``$SLURM_TMPDIR`` on the cluster's node in the folder ``/dataset`` inside the container with the option ``-B`` such that
each dataset downloaded by pytorch in ``/dataset`` will be available in ``$SLURM_TMPDIR``.

This will allow us to have code and scripts that are invariant to the cluster environment. The option ``-H`` specify what will be the container's home. For example,
if you have your code in ``$HOME/Project12345/Version35/`` you can specify ``-H $HOME/Project12345/Version35:/home``, thus the container will only have access to
the code inside ``Version35``.

If you want to run multiple commands inside the container you can use:

.. prompt:: bash $

        singularity exec --nv -H $HOME:/home -B $SLURM_TMPDIR:/dataset/ -B $SLURM_TMPDIR:/tmp_log/ -B $SCRATCH:/final_log/ $SLURM_TMPDIR/<YOUR_CONTAINER> bash -c 'pwd && ls && python <YOUR_CODE>'


Example: Interactive case (srun/salloc)
.......................................

Once you get an interactive session with slurm, copy ``<YOUR_CONTAINER>`` and ``<YOUR_DATASET>`` to ``$SLURM_TMPDIR``

.. prompt:: bash #,$ auto

        # 0. Get an interactive session
        $ srun --gres=gpu:1
        # 1. Copy your container on the compute node
        $ rsync -avz $SCRATCH/<YOUR_CONTAINER> $SLURM_TMPDIR
        # 2. Copy your dataset on the compute node
        $ rsync -avz $SCRATCH/<YOUR_DATASET> $SLURM_TMPDIR

then use ``singularity shell`` to get a shell inside the container

.. prompt:: bash #,$ auto

        # 3. Get a shell in your environment
        $ singularity shell --nv \
                -H $HOME:/home \
                -B $SLURM_TMPDIR:/dataset/ \
                -B $SLURM_TMPDIR:/tmp_log/ \
                -B $SCRATCH:/final_log/ \
                $SLURM_TMPDIR/<YOUR_CONTAINER>

.. prompt:: bash #,<Singularity_container>$ auto

        # 4. Execute your code
        <Singularity_container>$ python <YOUR_CODE>

**or** use ``singularity exec`` to execute ``<YOUR_CODE>``.

.. prompt:: bash #,$ auto

        # 3. Execute your code
        $ singularity exec --nv \
                -H $HOME:/home \
                -B $SLURM_TMPDIR:/dataset/ \
                -B $SLURM_TMPDIR:/tmp_log/ \
                -B $SCRATCH:/final_log/ \
                $SLURM_TMPDIR/<YOUR_CONTAINER> \
                python <YOUR_CODE>

You can create also the following alias to make your life easier.

.. prompt:: bash $

        alias my_env='singularity exec --nv \
                -H $HOME:/home \
                -B $SLURM_TMPDIR:/dataset/ \
                -B $SLURM_TMPDIR:/tmp_log/ \
                -B $SCRATCH:/final_log/ \
                $SLURM_TMPDIR/<YOUR_CONTAINER>'

This will allow you to run any code with:

.. prompt:: bash $

        my_env python <YOUR_CODE>


Example: sbatch case
....................

You can also create a ``sbatch`` script:

.. code-block:: bash
        :linenos:

        #!/bin/bash
        #SBATCH --cpus-per-task=6         # Ask for 6 CPUs
        #SBATCH --gres=gpu:1              # Ask for 1 GPU
        #SBATCH --mem=10G                 # Ask for 10 GB of RAM
        #SBATCH --time=0:10:00            # The job will run for 10 minutes

        # 1. Copy your container on the compute node
        rsync -avz $SCRATCH/<YOUR_CONTAINER> $SLURM_TMPDIR
        # 2. Copy your dataset on the compute node
        rsync -avz $SCRATCH/<YOUR_DATASET> $SLURM_TMPDIR
        # 3. Executing your code with singularity
        singularity exec --nv \
                -H $HOME:/home \
                -B $SLURM_TMPDIR:/dataset/ \
                -B $SLURM_TMPDIR:/tmp_log/ \
                -B $SCRATCH:/final_log/ \
                $SLURM_TMPDIR/<YOUR_CONTAINER> \
                python <YOUR_CODE>
        # 4. Copy whatever you want to save on $SCRATCH
        rsync -avz $SLURM_TMPDIR/<to_save> $SCRATCH


Issue with PyBullet and OpenGL libraries
........................................

If you are running certain gym environments that require ``pyglet``, you may encounter a problem when running your singularity instance with the Nvidia drivers using the ``--nv`` flag. This happens because the ``--nv`` flag also provides the OpenGL libraries:

.. code-block:: bash

        libGL.so.1 => /.singularity.d/libs/libGL.so.1
        libGLX.so.0 => /.singularity.d/libs/libGLX.so.0

If you don't experience those problems with ``pyglet``, you probably don't need to address this. Otherwise, you can resolve those problems by ``apt-get install -y libosmesa6-dev mesa-utils mesa-utils-extra libgl1-mesa-glx``, and then making sure that your ``LD_LIBRARY_PATH`` points to those libraries before the ones in ``/.singularity.d/libs``.

.. code-block:: bash

        %environment
                # ...
                export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa:$LD_LIBRARY_PATH


Mila cluster
............

On the Mila cluster ``$SCRATCH`` is not yet defined, you should add the experiment results you want to keep in ``/network/tmp1/$USER/``.
In order to use the sbatch script above and to match other cluster environment's names, you can define ``$SCRATCH`` as an alias for ``/network/tmp1/$USER`` with:

.. prompt:: bash $

        echo "export SCRATCH=/network/tmp1/$USER" >> ~/.bashrc

Then, you can follow the general procedure explained above.


Mila-cloud cluster
..................

On mila-cloud, the procedure is the same as above except that you have to define ``$SCRATCH`` as:

.. prompt:: bash $

        echo "export SCRATCH=/scratch/$USER" >> ~/.bashrc


Compute Canada
..............

Using singularity on Compute Canada is similar except that you need to add Yoshua's account name and load singularity.
Here is an example of a ``sbatch`` script using singularity on compute Canada cluster:

.. warning:: You should use singularity/2.6 or singularity/3.4. There is a bug in singularity/3.2 which makes gpu unusable.

.. code-block:: bash
        :linenos:

        #!/bin/bash
        #SBATCH --account=rpp-bengioy     # Yoshua pays for your job
        #SBATCH --cpus-per-task=6         # Ask for 6 CPUs
        #SBATCH --gres=gpu:1              # Ask for 1 GPU
        #SBATCH --mem=32G                 # Ask for 32 GB of RAM
        #SBATCH --time=0:10:00            # The job will run for 10 minutes
        #SBATCH --output="/scratch/<user>/slurm-%j.out" # Modify the output of sbatch

        # 1. You have to load singularity
        module load singularity
        # 2. Then you copy the container to the local disk
        rsync -avz $SCRATCH/<YOUR_CONTAINER> $SLURM_TMPDIR
        # 3. Copy your dataset on the compute node
        rsync -avz $SCRATCH/<YOUR_DATASET> $SLURM_TMPDIR
        # 4. Executing your code with singularity
        singularity exec --nv \
                -H $HOME:/home \
                -B $SLURM_TMPDIR:/dataset/ \
                -B $SLURM_TMPDIR:/tmp_log/ \
                -B $SCRATCH:/final_log/ \
                $SLURM_TMPDIR/<YOUR_CONTAINER> \
                python <YOUR_CODE>
        # 5. Copy whatever you want to save on $SCRATCH
        rsync -avz $SLURM_TMPDIR/<to_save> $SCRATCH
