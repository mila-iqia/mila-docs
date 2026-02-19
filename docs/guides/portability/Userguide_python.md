## Virtual environments

A virtual environment in Python is a local, isolated environment in which you
can install or uninstall Python packages without interfering with the global
environment (or other virtual environments). It usually lives in a directory
(location varies depending on whether you use venv, conda or poetry). In order
to use a virtual environment, you have to **activate** it. Activating an
environment essentially sets environment variables in your shell so that:

* `python` points to the right Python version for that environment (different
  virtual environments can use different versions of Python!)
* `python` looks for packages in the virtual environment
* `pip install` installs packages into the virtual environment
* Any shell commands installed via `pip install` are made available

To run experiments within a virtual environment, you can simply activate it
in the script given to `sbatch`.

### UV


In many cases, where your dependencies are Python packages, we highly recommend using `UV
<https://docs.astral.sh/uv>`__, a modern package manager for Python.

In addition to all the same features as pip, it also manages Python installations,
virtual environments, and makes your environments easier to reproduce and reuse across compute clusters.

!!! note
    UV is not currently available as a module on the Mila or DRAC clusters at the time of writing.
    To use it, you first need to install it using this command on a cluster login node:
    
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```


|                                          | Pip/virtualenv command                                                        | UV pip equivalent                                                | UV `project` command (recommended)                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| Create your virtualenv                   | `module load python/3.10`<br>then `python -m venv`                            | `uv venv`_                                                       | `uv init`_ and `uv sync`_                                          |
| Activate the virtualenv                  | `. .venv/bin/activate`                                                        | (same)                                                           | (same, but often unnecessary)                                      |
| Install a package                        | activate venv then `pip install`                                              | `uv pip install`_                                                | `uv add`_                                                          |
| Run a command     (ex. `python main.py`) | `module load python`, then<br>`. <venv>/bin/activate`, then  `python main.py` | `. <venv>/bin/activate`,<br>then `python main.py`                | `uv run python main.py`                                            |
| Where are dependencies declared?         | *Maybe* in a `requirements.txt`, `setup.py` or `pyproject.toml`               | *Maybe* in a `requirements.txt`,  `setup.py` or `pyproject.toml` | always in `pyproject.toml`                                         |
| Easy to change Python   versions?        | No                                                                            | somewhat                                                         | Yes: `uv python pin <version>` or<br> `uv sync --python <version>` |

While you can use UV as a drop-in replacement for pip, we recommend adopting a [project-based workflow](https://docs.astral.sh/uv/guides/projects/):

* Use [`uv init`](https://docs.astral.sh/uv/reference/cli/#uv-init) to create a new project. A [`pyproject.toml`](https://docs.astral.sh/uv/guides/projects/#pyprojecttoml) file will be created. This is where your dependencies are listed.
   
    ```console
    uv init --python=3.12
    ```

* Use [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add) to add (and [`uv remove`](https://docs.astral.sh/uv/reference/cli/#uv-remove) to remove) dependencies to your project. This will update the `pyproject.toml` file and update the virtual environment.
   
    ```console
    uv add torch
    ```

* Use [`uv run`](https://docs.astral.sh/uv/reference/cli/#uv-run) to run commands, for example `uv run python train.py`.
  This will automatically do the following:

    1.  Create or update the virtualenv (with the correct Python version) if necessary, based the dependencies in `pyproject.toml`.
    2. Activates the virtualenv.
    3. Runs the command you provided, e.g. `python train.py`.
   
    ```console
    uv run python main.py
    ```

### Pip/Virtualenv

Pip is the most widely used package manager for Python and each cluster provides
several Python versions through the associated module which comes with pip. In
order to install new packages, you will first have to create a personal space
for them to be stored.  The usual solution (as it is the recommended solution
on Digital Research Alliance of Canada clusters) is to use `virtual
environments <https://virtualenv.pypa.io/en/stable/>`_, although [using_uv](#using_uv) is now
the recommended way to manage Python installations, virtual environments and dependencies.

!!! note
    We recommend you use `UV <https://docs.astral.sh/uv>`_ to manage your Python
    virtual environments instead of doing it manually.
    [The previous section](#uv) gives an overview of how to install it and use it.

First, load the Python module you want to use:
```bash
module load python/3.10
```

Then, create a virtual environment in your `home` directory:
```bash
python -m venv $HOME/<env>
```

Where `<env>` is the name of your environment. Finally, activate the environment:

```bash
source $HOME/<env>/bin/activate
```

You can now install any Python package you wish using the `pip` command, e.g.
[pytorch ](https://pytorch.org/get-started/locally):

```bash
pip install torch torchvision
```

Or [Tensorflow ](https://www.tensorflow.org/install/gpu):
```bash
pip install tensorflow-gpu
```

### Conda

Another solution for Python is to use [miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [anaconda](httpshttps://docs.anaconda.com) which are also available through
the `module` command: (the use of Conda is not recommended for Digital Research
Alliance of Canada clusters due to the availability of custom-built packages for pip)

```bash
$ module load miniconda/3
[=== Module miniconda/3 loaded ===]
```

<!-- To enable conda environment functions, first use: -->

To create an environment (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for details) using a specific Python version, you may write:
```bash
conda create -n <env> python=3.9
```
Where `<env>` is the name of your environment. You can now activate it by doing:
```bash
conda activate <env>
```
You are now ready to install any Python package you want in this environment.
For instance, to install PyTorch, you can find the Conda command of any version
you want on [pytorch's website](https://pytorch.org/get-started/locally), e.g:

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

If you make a lot of environments and install/uninstall a lot of packages, it
can be good to periodically clean up Conda's cache:
```bash
conda clean -it
```

#### Mamba

When installing new packages with `conda install`, conda uses a built-in
dependency solver for solving the dependency graph of all packages (and their
versions) requested such that package dependency conflicts are avoided.

In some cases, especially when there are many packages already installed in a
conda environment, conda's built-in dependency solver can struggle to solve the
dependency graph, taking several to tens of minutes, and sometimes never
solving. In these cases, it is recommended to try [libmamba](https://conda.github.io/conda-libmamba-solver/getting-started/).

To install and set the `libmamba` solver, run the following commands:

```bash
# Install miniconda
# (you can not use the preinstalled anaconda/miniconda as installing libmamba
#  requires ownership over the anaconda/miniconda install directory)
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
$ bash Miniconda3-py310_22.11.1-1-Linux-x86_64.sh

# Install libmamba
$ conda install -n base conda-libmamba-solver
```

By default, conda uses the built-in solver when installing packages, even after
installing other solvers. To try `libmamba` once, add `--solver=libmamba` in
your ``conda install`` command. For example:
```bash
conda install tensorflow --solver=libmamba
```

You can set `libmamba` as the default solver by adding `solver: libmamba`
to your `.condarc` configuration file located under your `$HOME` directory.
You can create it if it doesn't exist. You can also run:
```bash
conda config --set solver libmamba
```