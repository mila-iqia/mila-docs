Monitoring
""""""""""

Every compute node on the Mila cluster has a monitoring daemon allowing you to
check the resource usage of your model and identify bottlenecks.
You can access the monitoring web page by typing in your browser: ``<node>.server.mila.quebec:19999``.

For example, if I have a job running on ``eos1`` I can type ``eos1.server.mila.quebec:19999`` and
the page below should appear.

.. image:: monitoring.png
    :align: center
    :alt: monitoring.png


Notable Sections
~~~~~~~~~~~~~~~~

You should focus your attention on the metrics below

* CPU
    * iowait (pink line): High values means your model is waiting on IO a lot (disk or network)

.. image:: monitoring_cpu.png
    :align: center
    :alt: monitoring_cpu.png

* RAM
    * Make sure you are only allocating enough to make your code run and not more otherwise you are wasting resources.

.. image:: monitoring_ram.png
    :align: center
    :alt: monitoring_ram.png

* NV
    * Usage of each GPU
    * You should make sure you use the GPU to its fullest
        * Select the biggest batch size if possible
        * Spawn multiple experiments

.. image:: monitoring_gpu.png
    :align: center
    :alt: monitoring_gpu.png

* Users:
    * In some cases the machine might seem slow, it may be useful to check if other people are using the machine as well

.. image:: monitoring_users.png
    :align: center
    :alt: monitoring_users.png
