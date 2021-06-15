.. sectnum::

Frequently asked questions (FAQs)
=================================


Connection/SSH
~~~~~~~~~~~~~~~

I'm getting ``connection refused`` while trying to connect to a login node
---------------------------------------------------------------------------

Login nodes are protected against brute force attacks and might ban your IP if it detects too many connections/failures.
You can try to unban yourself by using the following web page: https://unban.server.mila.quebec/



Shell
~~~~~

How do I change my shell
-------------------------

By default you will be assigned ``/bin/bash`` as a shell. If you would like to change for
another one, please submit a support ticket.



SLURM
~~~~~~


How can I get an interactive shell on the cluster ?
----------------------------------------------------

Use ``salloc [--slurm_options]`` without any executable at the end of the command, this will launch your
default shell on an interactive session. Remember that an interactive session is bound to the login node where
you start it so you could risk loosing your job if the login node becomes unreachable.


``srun: error: --mem and --mem-per-cpu are mutually exclusive`` error ?
------------------------------------------------------------------------

You can safely ignore this, ``salloc`` has a default memory flag in case you don't provide one.


How can I see where and if my jobs are running ?
-------------------------------------------------

Use ``squeue -u YOUR_USERNAME`` to see all your job status and locations.
To get more info on a running job, try ``scontrol show job #JOBID``


``Unable to allocate resources: Invalid account or account/partition combination specified``
--------------------------------------------------------------------------------------------

Chances are your account is not setup properly. You should file a ticket in our helpdesk: https://it-support.mila.quebec/ .


How do I cancel a job?
----------------------

Use the ``scancel #JOBID`` command with the job ID of the job you want cancelled. In the case you want
to cancel all your jobs, type ``scancel -u YOUR_USERNAME``. You can also cancel all your pending jobs for
instance with ``scancel -t PD``.

How can access a node on which one of my job is running ?
---------------------------------------------------------

You can ssh into a node on which you have a job running, your ssh connection will be adopted by your job, i.e.
if your job finishes your ssh connection will be automatically terminated. In order to connect to a node, you need to
have password-less ssh either with a key present in your home or with an ``ssh-agent``. You can generate a key on the
login node for password-less like this:


.. prompt:: bash $

    ssh-keygen (3xENTER)
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    chmod 700 ~/.ssh



I'm getting ``Permission denied (publickey)`` while trying to connect to a node ?
---------------------------------------------------------------------------------

See previous question



Where do I put my data during a job ?
-------------------------------------

Your ``/home`` as well as the datasets are on shared file-systems, it is recommended to copy them to the ``$SLURM_TMPDIR``
to better process them and leverage higher-speed local drives. If you run a low priority job subject to preemption, it's better
to keep any output you want to keep on the shared file systems because the ``$SLURM_TMPDIR`` is deleted at the end of each job.


I am getting the following error ``slurmstepd: error: Detected 1 oom-kill event(s) in step #####.batch cgroup.``
----------------------------------------------------------------------------------------------------------------

You exceeded the amount of memory allocated to your job, either you did not request enough memory or you have a
memory leak in your process. Try increasing the amount of memory requested with ``--mem=`` or ``--mem-per-cpu=``.


I am getting the following error ``fork: retry: Resource temporarily unavailable``
----------------------------------------------------------------------------------

You exceeded the limit of 2000 tasks/PIDs in your job, it probably means there is an issue
with a sub-process spawning too many processes
in your script. For any help with your software, please contact the helpdesk.
