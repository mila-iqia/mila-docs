Environment
============

Shared Folders
^^^^^^^^^^^^^^^

On each cloud provider, 2 important folders are available for your data and shared with every deployed compute nodes.

+----------------------------+---------------------------------------------------------+--------------+
| Path                       |       Description                                       |  Disk Size   |
+============================+=========================================================+==============+
| ``/home/mila/$USERNAME``   |  your own home, create your env and store your scripts  |              |
+----------------------------+---------------------------------------------------------+--------------+
| ``/scratch``               |  local to each cluster, to save your datasets/outputs   |              |
+----------------------------+---------------------------------------------------------+--------------+




.. Synchronization
.. ^^^^^^^^^^^^^^^
.. 
.. In order to keep your home in sync between the login node(s) of each cloud provider, a bi-directional syncing program
.. called ``unison`` has been configured to transfer securely (via ssh) your files between the login nodes.
.. 
.. .. note::
..   Only ``/home/mila/$USERNAME/`` is currently synced, ``/scratch`` is not replicated across regions and
..   for now is unique among a provider (1 scratch for Azure, 1 for GCP), you are encouraged to use it for
..   job outputs that do not need to be replicated.
.. 
.. To start a synchronization between login nodes, just type ``home_sync``
.. 
.. This script takes the same argument as ``unison`` and if it is started without any argument, it will be started with the ``-batch`` option.
.. If you want to test the connection, you can try
.. 
.. .. prompt:: bash $
.. 
..     home_sync -testServer
.. 
.. and to display the available commands
.. 
.. .. prompt:: bash $
.. 
..     home_sync -help
.. 
.. Configuration
.. -------------
.. 
.. A *default config* has been put in your private unison folder ``/home/mila/$USERNAME/.unison`` named ``mila-homes.prf``.
.. 
.. The default should be good as a starting point but if you have a lot of files stored inside your home folder,
.. you should add some filters to avoid syncing unnecessary files between login nodes or add rules to force hidden folders to be synced.
.. 
.. If you want to know more about the unison config, you can look at the `unison documentation <http://www.cis.upenn.edu/~bcpierce/unison/download/releases/stable/unison-manual.html>`_
.. 
.. 
.. .. .. note::
.. ..    FOR LATER:
.. ..
.. ..    Each job submission will trigger a sync between all nodes for your account so you might want to
.. ..    keep it quick in order for your job to start quickly
.. ..
.. ..    **DO NOT MODIFY THE CONFIG NAME/LOCATION as it is used to tell SLURM that your home has been synced and that jobs can run**
.. 
.. 
.. Backups
.. -------
.. 
.. By default, all overwritten/deleted files are backup-ed inside the ``.unison/backup/`` folder, you can change this behaviour or cherry-pick the file/folders to backup.



.. Workflow
.. --------
..
.. **NOT ACTIVE**
..
.. Each time you submit a job, unison will be started to synchronize files between login nodes,
.. your job will be delayed and the ``home_sync`` comment will be added to your job on each cluster.
..
.. The job will be replicated across clusters/providers but each job will be eligible to start only when the local sync is successful.
..
.. :Step-by-step:
..   1. ``sbatch/srun/salloc`` at time ``t1``
..   2. job is replicated on each cluster
..   3. on each cluster:
..     a. unison is started and finishes at time ``t2``
..     b. jobs submitted before ``t1`` are eligible to start
