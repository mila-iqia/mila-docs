Parts of a computing cluster
============================

In order to provide high performance computation capabilities, clusters can
combine hundreds to thousands of computers, called *nodes*, which are all
inter-connected with a high-performance communication network. Most nodes are
designed for high-performance computations, but clusters can also use
specialized nodes to offer parallel file systems, databases, login nodes and
even the cluster scheduling functionality as pictured in the image below.

.. image:: cluster_overview2.png


The Login Nodes
---------------

To execute computing processes on a cluster, you must first connect to a login
node. These login nodes are the entry point to most clusters. Another entry
point to some clusters such as the Mila cluster is the JupyterHub WEB
interface, but we'll read about that later.

For now let's return to the subject of this section; Login nodes. To connect to
these, you would typically use a remote shell connection. The most usual tool
to do so is SSH. You'll hear and read a lot about this tool. Imagine it as a
very long (and somewhat magical) extension cord which connects the computer you
are using now, such as your laptop, to a remote computer's terminal shell. If
you followed through with the tutorial in the section above, you already know
what a terminal shell is.

It is important to note here the difference in intended uses between the
compute nodes and the login nodes.

While the Compute Nodes are meant for heavy computation, the Login Nodes are
not. In the field of artificial intelligence, you will usually be on the hunt
for GPUs. The compute nodes are the ones with GPU capacity.

The login nodes however are used by everyone who uses the cluster and care must
be taken not to overburden these nodes. Consequently, only very short and light
processes should be run on these otherwise the cluster may become inaccessible.
In other words, please refrain from executing long or compute intensive
processes on login nodes because it affects all other users. In some cases, you
will also find that doing so might get you into trouble.
