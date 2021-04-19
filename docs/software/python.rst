.. _python:

Python
######

Pip/Virtualenv
"""""""""""""""

Pip is the preferred package manager for Python and each cluster provides several Python versions through the associated module
which comes with pip. In order to install new packages, you will first have to create a personal space for them to be stored.
The preferred solution (as it is the preferred solution on Compute Canada clusters) is
to use `virtual environments <https://virtualenv.pypa.io/en/stable/>`_.

First, load the python module you want to use:

.. prompt:: bash $

    module load python/3.6

Then, create a virtual environment in your ``home`` directory:

.. prompt:: bash $

    virtualenv $HOME/<env>

where ``<env>`` is the name of your environment. Finally, activate the environment:

.. prompt:: bash $

    source $HOME/<env>/bin/activate

You can now install any python package you wish using the ``pip`` command, e.g. `pytorch <https://pytorch.org/get-started/locally>`_:

.. prompt:: bash (<env>)$

    pip install torch torchvision

or `Tensorflow <https://www.tensorflow.org/install/gpu>`_:

.. prompt:: bash (<env>)$

    pip install tensorflow-gpu

Conda
"""""

Another solution for Python is to use `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
`anaconda <https://docs.anaconda.com>`_ which are also available through the ``module`` command:
(the use of conda is not recommended for Compute Canada Clusters due to the availability of custom-built packages for pip)

.. prompt:: bash $, auto

    $ module load miniconda/3
    [=== Module miniconda/3 loaded ===]
    To enable conda environment functions, first use:

    $ conda-activate

Then like advised, if you want to enable ``conda activate/deactivate`` functions, start the following alias once

.. prompt:: bash $

    conda-activate

To create an environment (see `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for details) do:

.. prompt:: bash $

    conda create -n <env> python=3.6

where ``<env>`` is the name of your environment. You can now activate it by doing:

.. prompt:: bash $

    conda activate <env>

You are now ready to install any python package you want in this environment.
For instance, to install pytorch, you can find the conda command of any version you want on `pytorch's website <https://pytorch.org/get-started/locally>`_, e.g:

.. prompt:: bash (<env>)$

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

Don't forget to clean the environment after each install:

.. prompt:: bash (<env>)$

    conda clean --all
