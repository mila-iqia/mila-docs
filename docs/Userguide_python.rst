.. _python:

Virtual environments
====================

A virtual environment in Python is a local, isolated environment in which you
can install or uninstall Python packages without interfering with the global
environment (or other virtual environments). It usually lives in a directory
(location varies depending on whether you use venv, conda or poetry). In order
to use a virtual environment, you have to **activate** it. Activating an
environment essentially sets environment variables in your shell so that:

* ``python`` points to the right Python version for that environment (different
  virtual environments can use different versions of Python!)
* ``python`` looks for packages in the virtual environment
* ``pip install`` installs packages into the virtual environment
* Any shell commands installed via ``pip install`` are made available

To run experiments within a virtual environment, you can simply activate it
in the script given to ``sbatch``.


Pip/Virtualenv
--------------

Pip is the most widely used package manager for Python and each cluster provides
several Python versions through the associated module which comes with pip. In
order to install new packages, you will first have to create a personal space
for them to be stored.  The usual solution (as it is the recommended solution
on Digital Research Alliance of Canada clusters) is to use `virtual
environments <https://virtualenv.pypa.io/en/stable/>`_, although :ref:`uv` is now
the recommended way to manage Python installations, virtual environments and dependencies.

.. note::
   We highly recommend you use `UV <https://docs.astral.sh/uv>`_ to manage your Python
   virtual environments instead of doing it manually.
   :ref:`The next section <uv>` will give an overview of how to install it and use it.

First, load the Python module you want to use:

.. prompt:: bash $

   module load python/3.8

Then, create a virtual environment in your ``home`` directory:

.. prompt:: bash $

   python -m venv $HOME/<env>

Where ``<env>`` is the name of your environment. Finally, activate the environment:

.. prompt:: bash $

   source $HOME/<env>/bin/activate

You can now install any Python package you wish using the ``pip`` command, e.g.
`pytorch <https://pytorch.org/get-started/locally>`_:

.. prompt:: bash (<env>)$

   pip install torch torchvision

Or `Tensorflow <https://www.tensorflow.org/install/gpu>`_:

.. prompt:: bash (<env>)$

   pip install tensorflow-gpu

UV
--

In many cases, where your dependencies are Python packages, we highly recommend using `UV
<https://docs.astral.sh/uv>`_.

UV is a modern package manager for Python. In addition to all the same features as pip,
it also manages Python installations, virtual environments, and makes your environments
easier to reproduce and reuse across compute clusters.

To install UV, follow the instructions at `this link <https://docs.astral.sh/uv/getting-started/installation/>`_.


+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
|                         | Pip/virtualenv command             | UV pip equivalent                  | UV `project`_ command (recommended) | 
+=========================+====================================+====================================+=====================================+
| Create your virtualenv  | ``module load python/3.10``        | `uv venv`_                         | `uv init`_ and `uv sync`_           |
|                         | then ``python -m venv``            |                                    |                                     |
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
| Activate the virtualenv | ``. .venv/bin/activate``           | (same)                             | (same, but often unnecessary)       |
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
| Install a package       | activate venv then ``pip install`` | `uv pip install`_                  | `uv add`_                           |    
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
| Run a command           | ``module load python``, then       |                                    |                                     |
| (ex. ``python main.py``)| ``. <venv>/bin/activate``, then    | ``. <venv>/bin/activate``,         |                                     |
|                         | ``python main.py``                 | then ``python main.py``            | ``uv run python main.py``           |
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
| Where are               | *Maybe* in a ``requirements.txt``, | *Maybe* in a ``requirements.txt``, | `pyproject.toml`_                   |
| dependencies declared?  | ``setup.py`` or ``pyproject.toml`` | ``setup.py`` or ``pyproject.toml`` |                                     |
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+
| Easy to change Python   | No                                 | somewhat                           | Yes                                 |
| versions?               |                                    |                                    |                                     |
+-------------------------+------------------------------------+------------------------------------+-------------------------------------+


While you can use UV as a drop-in replacement for pip, we recommend adopting a `project-based workflow`_:


* Use `uv init`_ to create a new project. A ``pyproject.toml`` file will be created. This is where your dependencies are listed.

   .. prompt:: bash $

      uv init --python=3.12

* Use `uv add`_ (and `uv remove <https://docs.astral.sh/uv/reference/cli/#uv-remove>`_ to remove) dependencies to your project. This will update the ``pyproject.toml`` file and update the virtual environment.

   .. prompt:: bash $

      uv add torch

* Use `uv run`_ to run commands, for example ``uv run python train.py``. This will automatically do the following:
   1. Create or update the virtualenv (with the correct Python version) if necessary, based the dependencies in ``pyproject.toml``.
   2. Activates the virtualenv.
   3. Runs the command you provided, e.g. ``python train.py``.

   .. prompt:: bash $

      uv run python main.py


.. _project-based workflow: https://docs.astral.sh/uv/guides/projects/
.. _pyproject.toml: https://docs.astral.sh/uv/guides/projects/#pyprojecttoml
.. _project: https://docs.astral.sh/uv/guides/projects/
.. _uv init: https://docs.astral.sh/uv/reference/cli/#uv-init
.. _uv add: https://docs.astral.sh/uv/reference/cli/#uv-add
.. _uv remove: https://docs.astral.sh/uv/reference/cli/#uv-remove
.. _uv run: https://docs.astral.sh/uv/reference/cli/#uv-run
.. _uv sync: https://docs.astral.sh/uv/reference/cli/#uv-sync
.. _uv pip install: https://docs.astral.sh/uv/reference/cli/#uv-pip-install
.. _uv venv: https://docs.astral.sh/uv/reference/cli/#uv-venv

Conda
-----

Another solution for Python is to use `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda
<https://docs.anaconda.com>`_ which are also available through the ``module``
command: (the use of Conda is not recommended for Digital Research Alliance of
Canada clusters due to the availability of custom-built packages for pip)

.. prompt:: bash $, auto

   $ module load miniconda/3
   [=== Module miniconda/3 loaded ===]
   To enable conda environment functions, first use:

To create an environment (see `here
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for details) using a specific Python version, you may write:

.. prompt:: bash $

   conda create -n <env> python=3.9

Where ``<env>`` is the name of your environment. You can now activate it by doing:

.. prompt:: bash $

   conda activate <env>

You are now ready to install any Python package you want in this environment.
For instance, to install PyTorch, you can find the Conda command of any version
you want on `pytorch's website <https://pytorch.org/get-started/locally>`_, e.g:

.. prompt:: bash (<env>)$

   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

If you make a lot of environments and install/uninstall a lot of packages, it
can be good to periodically clean up Conda's cache:

.. prompt:: bash (<env>)$

   conda clean -it

Mamba
^^^^^

When installing new packages with ``conda install``, conda uses a built-in
dependency solver for solving the dependency graph of all packages (and their
versions) requested such that package dependency conflicts are avoided.

In some cases, especially when there are many packages already installed in a
conda environment, conda's built-in dependency solver can struggle to solve the
dependency graph, taking several to tens of minutes, and sometimes never
solving. In these cases, it is recommended to try `libmamba
<https://conda.github.io/conda-libmamba-solver/getting-started/>`_.

To install and set the ``libmamba`` solver, run the following commands:

.. prompt:: bash $, auto

   \# Install miniconda
   \# (you can not use the preinstalled anaconda/miniconda as installing libmamba
   \#  requires ownership over the anaconda/miniconda install directory)
   $ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
   $ bash Miniconda3-py310_22.11.1-1-Linux-x86_64.sh

   \# Install libmamba
   $ conda install -n base conda-libmamba-solver

By default, conda uses the built-in solver when installing packages, even after
installing other solvers. To try ``libmamba`` once, add ``--solver=libmamba`` in
your ```conda install``` command. For example:

.. prompt:: bash $

   conda install tensorflow --solver=libmamba

You can set ``libmamba`` as the default solver by adding ``solver: libmamba``
to your ``.condarc`` configuration file located under your ``$HOME`` directory.
You can create it if it doesn't exist. You can also run:

.. prompt:: bash $

   conda config --set solver libmamba
