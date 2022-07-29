.. _Using containers:

Using containers on clusters
----------------------------


How to use containers on clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On every cluster with Slurm, datasets and intermediate results should go in
``$SLURM_TMPDIR`` while the final experiment results should go in ``$SCRATCH``.
In order to use the container you built, you need to copy it on the cluster you
want to use.

.. warning:: You should always store your container in $SCRATCH !

Then reserve a node with srun/sbatch, copy the container and your dataset on the
node given by SLURM (i.e in ``$SLURM_TMPDIR``) and execute the code
``<YOUR_CODE>`` within the container ``<YOUR_CONTAINER>`` with:

.. prompt:: bash $

        singularity exec --nv -H $HOME:/home -B $SLURM_TMPDIR:/dataset/ -B $SLURM_TMPDIR:/tmp_log/ -B $SCRATCH:/final_log/ $SLURM_TMPDIR/<YOUR_CONTAINER> python <YOUR_CODE>


Remember that ``/dataset``, ``/tmp_log`` and ``/final_log`` were created in the
previous section. Now each time, we'll use singularity, we are explicitly
telling it to mount ``$SLURM_TMPDIR`` on the cluster's node in the folder
``/dataset`` inside the container with the option ``-B`` such that each dataset
downloaded by PyTorch in ``/dataset`` will be available in ``$SLURM_TMPDIR``.

This will allow us to have code and scripts that are invariant to the cluster
environment. The option ``-H`` specify what will be the container's home. For
example, if you have your code in ``$HOME/Project12345/Version35/`` you can
specify ``-H $HOME/Project12345/Version35:/home``, thus the container will only
have access to the code inside ``Version35``.

If you want to run multiple commands inside the container you can use:

.. prompt:: bash $

        singularity exec --nv -H $HOME:/home -B $SLURM_TMPDIR:/dataset/ \
           -B $SLURM_TMPDIR:/tmp_log/ -B $SCRATCH:/final_log/ \
           $SLURM_TMPDIR/<YOUR_CONTAINER> bash -c 'pwd && ls && python <YOUR_CODE>'


Example: Interactive case (srun/salloc)
"""""""""""""""""""""""""""""""""""""""

Once you get an interactive session with SLURM, copy ``<YOUR_CONTAINER>`` and
``<YOUR_DATASET>`` to ``$SLURM_TMPDIR``

.. prompt:: bash #,$ auto

        # 0. Get an interactive session
        $ srun --gres=gpu:1
        # 1. Copy your container on the compute node
        $ rsync -avz $SCRATCH/<YOUR_CONTAINER> $SLURM_TMPDIR
        # 2. Copy your dataset on the compute node
        $ rsync -avz $SCRATCH/<YOUR_DATASET> $SLURM_TMPDIR

Then use ``singularity shell`` to get a shell inside the container

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
""""""""""""""""""""

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
           python "<YOUR_CODE>"
   # 4. Copy whatever you want to save on $SCRATCH
   rsync -avz $SLURM_TMPDIR/<to_save> $SCRATCH


Issue with PyBullet and OpenGL libraries
""""""""""""""""""""""""""""""""""""""""

If you are running certain gym environments that require ``pyglet``, you may
encounter a problem when running your singularity instance with the Nvidia
drivers using the ``--nv`` flag. This happens because the ``--nv`` flag also
provides the OpenGL libraries:

.. code-block:: bash

   libGL.so.1 => /.singularity.d/libs/libGL.so.1
   libGLX.so.0 => /.singularity.d/libs/libGLX.so.0

If you don't experience those problems with ``pyglet``, you probably don't need
to address this. Otherwise, you can resolve those problems by ``apt-get install
-y libosmesa6-dev mesa-utils mesa-utils-extra libgl1-mesa-glx``, and then making
sure that your ``LD_LIBRARY_PATH`` points to those libraries before the ones in
``/.singularity.d/libs``.

.. code-block:: bash

   %environment
           # ...
           export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa:$LD_LIBRARY_PATH


Mila cluster
""""""""""""

On the Mila cluster ``$SCRATCH`` is not yet defined, you should add the
experiment results you want to keep in ``/network/scratch/<u>/<username>/``. In
order to use the sbatch script above and to match other cluster environment's
names, you can define ``$SCRATCH`` as an alias for
``/network/scratch/<u>/<username>`` with:

.. prompt:: bash $

   echo "export SCRATCH=/network/scratch/${USER:0:1}/$USER" >> ~/.bashrc

Then, you can follow the general procedure explained above.



Compute Canada
""""""""""""""

Using singularity on Compute Canada is similar except that you need to add
Yoshua's account name and load singularity.  Here is an example of a ``sbatch``
script using singularity on compute Canada cluster:

.. warning:: You should use singularity/2.6 or singularity/3.4. There is a bug
   in singularity/3.2 which makes gpu unusable.

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
           python "<YOUR_CODE>"
   # 5. Copy whatever you want to save on $SCRATCH
   rsync -avz $SLURM_TMPDIR/<to_save> $SCRATCH
