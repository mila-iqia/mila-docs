.. _modules:

Modules
#######

Mila and Compute Canada provides various software (such as python, cuda, etc) through the ``module`` command.
Modules are small files which modify your environment variables (PATH, LD_LIBRARY_PATH, etc...) to register the correct
location of the software you wish to use.


The ``module`` command
"""""""""""""""""""""""

For a list of available modules, simply use:

.. prompt:: bash $, auto

    $ module avail
    --------------------------------------------------------------------------------------------------------------- Global Aliases ---------------------------------------------------------------------------------------------------------------
       cuda/10.0 -> cudatoolkit/10.0    cuda/9.2      -> cudatoolkit/9.2                                 pytorch/1.4.1       -> python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.4.1    tensorflow/1.15 -> python/3.7/tensorflow/1.15
       cuda/10.1 -> cudatoolkit/10.1    mujoco-py     -> python/3.7/mujoco-py/2.0                        pytorch/1.5.0       -> python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0    tensorflow/2.2  -> python/3.7/tensorflow/2.2
       cuda/10.2 -> cudatoolkit/10.2    mujoco-py/2.0 -> python/3.7/mujoco-py/2.0                        pytorch/1.5.1       -> python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.1
       cuda/11.0 -> cudatoolkit/11.0    pytorch       -> python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.1    tensorflow          -> python/3.7/tensorflow/2.2
       cuda/9.0  -> cudatoolkit/9.0     pytorch/1.4.0 -> python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.4.0    tensorflow-cpu/1.15 -> python/3.7/tensorflow/1.15

    --------------------------------------------------------------------------------------------------- /cvmfs/config.mila.quebec/modules/Core ---------------------------------------------------------------------------------------------------
       Mila       (S,L)    anaconda/3 (D)    go/1.13.5        miniconda/2        mujoco/1.50        python/2.7    python/3.6        python/3.8           singularity/3.0.3    singularity/3.2.1    singularity/3.5.3 (D)
       anaconda/2          go/1.12.4         go/1.14   (D)    miniconda/3 (D)    mujoco/2.0  (D)    python/3.5    python/3.7 (D)    singularity/2.6.1    singularity/3.1.1    singularity/3.4.2

    ------------------------------------------------------------------------------------------------- /cvmfs/config.mila.quebec/modules/Compiler -------------------------------------------------------------------------------------------------
       python/3.7/mujoco-py/2.0

    --------------------------------------------------------------------------------------------------- /cvmfs/config.mila.quebec/modules/Cuda ---------------------------------------------------------------------------------------------------
       cuda/10.0/cudnn/7.3        cuda/10.0/nccl/2.4         cuda/10.1/nccl/2.4     cuda/11.0/nccl/2.7        cuda/9.0/nccl/2.4     cudatoolkit/9.0     cudatoolkit/10.1        cudnn/7.6/cuda/10.0/tensorrt/7.0
       cuda/10.0/cudnn/7.5        cuda/10.1/cudnn/7.5        cuda/10.2/cudnn/7.6    cuda/9.0/cudnn/7.3        cuda/9.2/cudnn/7.6    cudatoolkit/9.2     cudatoolkit/10.2        cudnn/7.6/cuda/10.1/tensorrt/7.0
       cuda/10.0/cudnn/7.6 (D)    cuda/10.1/cudnn/7.6 (D)    cuda/10.2/nccl/2.7     cuda/9.0/cudnn/7.5 (D)    cuda/9.2/nccl/2.4     cudatoolkit/10.0    cudatoolkit/11.0 (D)    cudnn/7.6/cuda/9.0/tensorrt/7.0

    ------------------------------------------------------------------------------------------------- /cvmfs/config.mila.quebec/modules/Pytorch --------------------------------------------------------------------------------------------------
       python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.4.1    python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.1 (D)    python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0
       python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0    python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.4.1        python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.1 (D)

    ------------------------------------------------------------------------------------------------ /cvmfs/config.mila.quebec/modules/Tensorflow ------------------------------------------------------------------------------------------------
       python/3.7/tensorflow/1.15    python/3.7/tensorflow/2.0    python/3.7/tensorflow/2.2 (D)



Modules can be loaded using the ``load`` command:

.. prompt:: bash $

    module load <module>


To search for a module or a software, use the command ``spider``:


.. prompt:: bash $

    module spider search_term

E.g.: by default, ``python2`` will refer to the os-shipped installation of ``python2.7`` and ``python3`` to ``python3.6``.
If you want to use ``python3.7`` you can type:


.. prompt:: bash $

   module load python3.7



Available Software
"""""""""""""""""""""

Modules are divided in 5 main sections:

============================= =========================================================================================================
Section                          Description
============================= =========================================================================================================
Core                             Base interpreter and software (Python, go, etc...)
Compiler                         Interpreter-dependent software (*see the note below*)
Cuda                             Toolkits, cudnn and related libraries
Pytorch/Tensorflow               Pytorch/TF built with a specific Cuda/Cudnn
                                 version for Mila's GPUs (*see the related paragraph*)
============================= =========================================================================================================


.. note::

	Modules which are nested (../../..) usually depend on other software/module loaded alongside the main module.
	No need to load the dependent software, the complex naming scheme allows an automatic detection of the dependent module(s):

	i.e.:
	Loading ``cudnn/7.6/cuda/9.0/tensorrt/7.0`` will load ``cudnn/7.6`` and ``cuda/9.0`` alongside

	``python/3.X`` is a particular dependency which can be served through ``python/3.X`` or ``anaconda/3`` and is not automatically
	loaded to let the user pick his favorite flavor.




Default package location
""""""""""""""""""""""""""


Python by default uses the user site package first and packages provided by ``module`` last to not interfere with your installation.
If you want to skip packages installed in your site package (in your /home folder), you have to start Python with the ``-s`` flag.

To check which package is loaded at import, you can print ``package.__file__`` to get the full path of the package.

*Example:*

.. prompt:: bash $, auto

    $ module load pytorch/1.5.0
    $ python -c 'import torch;print(torch.__file__)'
    /home/mila/my_home/.local/lib/python3.7/site-packages/torch/__init__.py   <== package from your own site-package

Now with the ``-s`` flag:

.. prompt:: bash $, auto

    $ module load pytorch/1.5.0
    $ python -s -c 'import torch;print(torch.__file__)'
    /cvmfs/ai.mila.quebec/apps/x86_64/debian/pytorch/python3.7-cuda10.1-cudnn7.6-v1.5.0/lib/python3.7/site-packages/torch/__init__.py'
