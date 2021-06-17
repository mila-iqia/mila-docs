Overview
--------

What is Singularity?
^^^^^^^^^^^^^^^^^^^^

Running Docker on SLURM is a security problem (e.g. running as root, being able to mount any directory).
The alternative is to use Singularity, which is a popular solution in the world of HPC.

There is a good level of compatibility between Docker and Singularity,
and we can find many exaggerated claims about able to convert containers 
from Docker to Singularity without any friction.
Oftentimes, Docker images from dockerhub are 100% compatible with Singularity,
and they can indeed be used without friction, but things get messy when
we try to convert our own Docker build files to Singularity recipes.

Links to official documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* official `Singularity user guide <https://singularity-docs.readthedocs.io/en/latest/>`_ (this is the one you will use most often)
* official `Singularity admin guide <https://sylabs.io/guides/latest/admin-guide/>`_

Overview of the steps used in practice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most often, the process to create and use a Singularity container is:

* on your linux computer (at home or work)

  * select a Docker image from dockerhub (e.g. pytorch/pytorch)
  * make a recipe file for Singularity that starts with that dockerhub image
  * build the recipe file, thus creating the image file (e.g. ``my-pytorch-image.sif``)
  * test your singularity container before send it over to the cluster
  * ``rsync -av my-pytorch-image.sif <login-node>:Documents/my-singularity-images``

* on the login node for that cluster

  * queue your jobs with ``sbatch ...``
  * (note that your jobs will copy over the ``my-pytorch-image.sif`` to $SLURM_TMPDIR
    and will then launch Singularity with that image)
  * do something else while you wait for them to finish
  * queue more jobs with the same ``my-pytorch-image.sif``,
    reusing it many times over

In the following sections you will find specific examples or tips to accomplish
in practice the steps highlighted above.

Nope, not on MacOS
^^^^^^^^^^^^^^^^^^

Singularity does not work on MacOS, as of the time of this writing in 2021.
Docker does not *actually* run on MacOS, but there Docker silently installs a
virtual machine running Linux, which makes it a pleasant experience,
and the user does not need to care about the details of how Docker does it.
  
Given its origins in HPC, Singularity does not provide that kind of seamless
experience on MacOS, even though it's technically possible to run it
inside a Linux virtual machine on MacOS.

Where to build images
^^^^^^^^^^^^^^^^^^^^^

Building Singularity images is a rather heavy task, which can take 20 minutes
if you have a lot of steps in your recipe. This makes it a bad task to run on
the login nodes of our clusters, especially if it needs to be run regularly.

On the Mila cluster, we are lucky to have unrestricted internet access on the compute
nodes, which means that anyone can request an interactive CPU node (no need for GPU)
and build their images there without problem.

.. warning::
  Do not build Singularity images from scratch every time your run a job in a large batch.
  This will be a colossal waste of GPU time as well as internet bandwidth.
  If you setup your workflow properly (e.g. using bind paths for your code and data),
  you can spend months reusing the same Singularity image ``my-pytorch-image.sif``.


