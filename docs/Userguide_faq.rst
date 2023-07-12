Frequently asked questions (FAQs)
=================================


Connection/SSH issues
---------------------

I'm getting ``connection refused`` while trying to connect to a login node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Login nodes are protected against brute force attacks and might ban your IP if
it detects too many connections/failures. You will be automatically unbanned
after 1 hour. For any further problem, please `submit a support ticket.
<https://mila-iqia.atlassian.net/servicedesk/customer/portals>`_


Shell issues
------------

How do I change my shell ?
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default you will be assigned ``/bin/bash`` as a shell. If you would like to
change for another one, please `submit a support ticket.
<https://mila-iqia.atlassian.net/servicedesk/customer/portals>`_



SLURM issues
------------


How can I get an interactive shell on the cluster ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``salloc [--slurm_options]`` without any executable at the end of the
command, this will launch your default shell on an interactive session. Remember
that an interactive session is bound to the login node where you start it so you
could risk losing your job if the login node becomes unreachable.


How can I reset my cluster password ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reset your password, please `submit a support ticket.
<https://mila-iqia.atlassian.net/servicedesk/customer/portals>`_

**Warning**: your cluster password is the same as your Google Workspace account. So,
after reset, you must use the new password for all your Google services.

srun: error: --mem and --mem-per-cpu are mutually exclusive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can safely ignore this, ``salloc`` has a default memory flag in case you
don't provide one.


How can I see where and if my jobs are running ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``squeue -u YOUR_USERNAME`` to see all your job status and locations.
To get more info on a running job, try ``scontrol show job #JOBID``


Unable to allocate resources: Invalid account or account/partition combination specified
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chances are your account is not setup properly. You should `submit a support ticket.
<https://mila-iqia.atlassian.net/servicedesk/customer/portals>`_


How do I cancel a job?
^^^^^^^^^^^^^^^^^^^^^^

* To cancel a specific job, use ``scancel #JOBID``
* To cancel all your jobs (running and pending), use ``scancel -u YOUR_USERNAME``
* To cancel all your pending jobs only, use ``scancel -t PD``

How can I access a node on which one of my jobs is running ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can ssh into a node on which you have a job running, your ssh connection
will be adopted by your job, i.e.  if your job finishes your ssh connection will
be automatically terminated. In order to connect to a node, you need to have
password-less ssh either with a key present in your home or with an
``ssh-agent``. You can generate a key on the login node like this:


.. prompt:: bash $

    ssh-keygen (3xENTER)
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    chmod 700 ~/.ssh



I'm getting ``Permission denied (publickey)`` while trying to connect to a node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See previous question



Where do I put my data during a job ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your ``/home`` as well as the datasets are on shared file-systems, it is
recommended to copy them to the ``$SLURM_TMPDIR`` to better process them and
leverage higher-speed local drives. If you run a low priority job subject to
preemption, it's better to save any output you want to keep on the shared file
systems, because the ``$SLURM_TMPDIR`` is deleted at the end of each job.


slurmstepd: error: Detected 1 oom-kill event(s) in step #####.batch cgroup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You exceeded the amount of memory allocated to your job, either you did not
request enough memory or you have a memory leak in your process. Try increasing
the amount of memory requested with ``--mem=`` or ``--mem-per-cpu=``.


fork: retry: Resource temporarily unavailable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You exceeded the limit of 2000 tasks/PIDs in your job, it probably means there
is an issue with a sub-process spawning too many processes in your script. For
any help with your software, please `submit a support ticket.
<https://mila-iqia.atlassian.net/servicedesk/customer/portals>`_


PyTorch issues
--------------


I randomly get ``INTERNAL ASSERT FAILED at "../aten/src/ATen/MapAllocator.cpp":263``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are using PyTorch 1.10.x and hitting `#67864
<https://github.com/pytorch/pytorch/issues/67864>`_,
for which the solution is `PR #72232
<https://github.com/pytorch/pytorch/pull/72232>`_
merged in PyTorch 1.11.x. For an immediate fix, consider the following compilable Gist:
`hack.cpp
<https://gist.github.com/obilaniu/b133470cb70410d841faca819d3921e5>`_.
Compile the patch to ``hack.so`` and then ``export LD_PRELOAD=/absolute/path/to/hack.so``
before executing the Python process that ``import torch`` a broken PyTorch 1.10.

For Hydra users who are using the submitit launcher plug-in, the ``env_set`` key cannot
be used to set ``LD_PRELOAD`` in the environment as it does so too late at runtime. The
dynamic loader reads ``LD_PRELOAD`` only once and very early during the startup of any
process, before the variable can be set from inside the process. The hack must therefore
be injected using the ``setup`` key in Hydra YAML config file::

  hydra:
    launcher:
      setup:
        - export LD_PRELOAD=/absolute/path/to/hack.so


On MIG GPUs, I get ``torch.cuda.device_count() == 0`` despite ``torch.cuda.is_available()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are using PyTorch 1.13.x and hitting `#90543
<https://github.com/pytorch/pytorch/issues/90543>`_,
for which the solution is `PR #92315
<https://github.com/pytorch/pytorch/pull/92315>`_
merged in PyTorch 2.0.

To avoid thus problem, update to PyTorch 2.0. If PyTorch 1.13.x is required, a
workaround is to add the following to your script:

.. code:: bash

   unset CUDA_VISIBLE_DEVICES

But this is no longer necessary with PyTorch >= 2.0.


I am told my PyTorch job abuses the filesystem with extreme amounts of IOPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fairly common issue in PyTorch is:

.. code:: none

   RuntimeError: one of the variables needed for gradient computation has been
   modified by an inplace operation: [torch.cuda.FloatTensor [1, 50, 300]],
   which is output 0 of SplitBackward, is at version 2; expected version 0
   instead. Hint: enable anomaly detection to find the operation that failed to
   compute its gradient, with torch.autograd.set_detect_anomaly(True).

PyTorch's autograd engine contains an "anomaly detection mode", which detects
such things as NaN/infinities being created, and helps debugging in-place
Tensor modifications. It is activated with

.. code:: python

   torch.autograd.set_detect_anomaly(True)

PyTorch's implementation of the anomaly-detection mode tracks where every Tensor
was created in the program. This involves the collection of the backtrace at the
point the Tensor was created.

**Unfortunately**, the collection of a backtrace involves a ``stat()`` system
call to **every** source file in the backtrace. This is considered a metadata
access to ``$HOME`` and results in intolerably heavy traffic to the shared
filesystem containing the source code, usually ``$HOME``, *whatever the location
of the dataset*, and *even if it is on* ``$SLURM_TMPDIR``. It is the source-code
files being polled, not the dataset. As there can be hundreds of PyTorch tensors
created per iteration and thousands of iterations per second, this mode results
in **extreme** amounts of IOPS to the filesystem.

.. warning::

    - **Do not use** ``torch.autograd.set_detect_anomaly(True)`` except for
      debugging an individual job interactively, and switch it off as soon as
      done using it.

    - **Do not set** ``torch.autograd.set_detect_anomaly(True)`` enabled
      unconditionally in all your jobs. **It is not a consequence-free aid**.
      Due to heavy use of filesystem calls, **it has a performance impact and
      slows down your code**, *on top* of abusing the filesystem.

    - **You will be contacted** if you violate these guidelines due to the
      severity of its impact on shared filesystems.
