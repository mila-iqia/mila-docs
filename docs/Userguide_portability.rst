Portability concerns and solutions
==================================

When working on a software project, it is important to be aware of all the
software and libraries the project relies on and to list them explicitly and
*under a version control system* in such a way that they can easily be
installed and made available on different systems. The upsides are significant:

* Easily install and run on the cluster
* Ease of collaboration
* Better reproducibility

To achieve this, try to always keep in mind the following aspects:

* **Versions:** For each dependency, make sure you have some record of the
  specific version you are using during development. That way, in the future, you
  will be able to reproduce the original environment which you know to be
  compatible. Indeed, the more time passes, the more likely it is that newer
  versions of some dependency have breaking changes. The ``pip freeze`` command can create
  such a record for Python dependencies.
* **Isolation:** Ideally, each of your software projects should be isolated from
  the others. What this means is that updating the environment for project A
  should *not* update the environment for project B. That way, you can freely
  install and upgrade software and libraries for the former without worrying about
  breaking the latter (which you might not notice until weeks later, the next time
  you work on project B!) Isolation can be achieved using :ref:`Python Virtual
  environments` and :ref:`containers`.

.. Creating a list of your software's dependencies
.. -----------------------------------------------
.. TODO


Managing your environments
--------------------------

.. include:: Userguide_python.rst


Using Modules
-------------

A lot of software, such as Python and Conda, is already compiled and available on
the cluster through the ``module`` command and its sub-commands. In particular,
if you wish to use ``Python 3.7`` you can simply do:

.. prompt:: bash $

    module load python/3.7

.. include:: Userguide_portability_modules.rst


On using containers
-------------------

Another option for creating portable code is :ref:`Using containers`.

Containers are a popular approach at deploying applications by packaging a lot
of the required dependencies together. The most popular tool for this is
`Docker <https://www.docker.com/>`_, but Docker cannot be used on the Mila
cluster (nor the other clusters from Compute Canada).

One popular mechanism for containerisation on a computational cluster is called
`Singularity <https://singularity-docs.readthedocs.io/en/latest/>`_.
This is the recommended approach for running containers on the
Mila cluster. See section :ref:`Singularity` for more details.
