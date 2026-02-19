# Quick Start


At Mila, you are given access to one or more clusters in order to run your experiments. To be very general, a cluster is a set of several computers working together to be accessed as a single system. Thus, it provides shared computing resources among the researchers to help them running their experiments. You can have more details about clusters on the page [What is a computer cluster?](../../Theory_cluster_parts.md).


## Cluster access

{%
    include-markdown "guides/quick_start/Userguide_cluster_access.md"
%}


## mila code

It is recommended to install [milatools](https://github.com/mila-iqia/milatools)
which will help in the set up of the SSH configuration needed to securely and
easily connect to the cluster. `milatools` also makes it easy to run and debug
code on the Mila cluster.

First you need to set up your SSH configuration using `mila init`. The
initialization of the SSH configuration is explained on the
[milatools README](https://github.com/mila-iqia/milatools#mila-init).

Once that is done, you may run [VSCode](https://code.visualstudio.com/) on the
cluster simply by using the
[Remote-SSH extension](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host)
and selecting `mila-cpu` as the host (in step 2).

`mila-cpu` allocates a single CPU and 8 GB of RAM. If you need more resources
from within VSCode (e.g. to run an ML model in a notebook), then you can use
`mila code`. For example, if you want a GPU, 32G of RAM and 4 cores, run this
command in the terminal:

```bash
mila code path/on/cluster --alloc --gres=gpu:1 --mem=32G -c 4
```

The details of the command can be found in the
[milatools README](https://github.com/mila-iqia/milatools#mila-code).
Remember that you need to first set up your SSH configuration using `mila init`
before the `mila code` command can be used.

## Using a Terminal

While VSCode provides a graphical interface for writing and debugging code on
the cluster, working on the cluster will require using a terminal to navigate
the filesystem, run commands, and manage jobs.

To open a terminal session on the cluster, connect using:

```bash
ssh mila
```

This will connect you to a login node where you can run commands, submit jobs,
and navigate the cluster filesystem.

## Next Steps

Once you have access to the cluster, you may want to:

* **Set up a framework**: For a quick example of setting up PyTorch on the
	cluster, see [PyTorch Setup](examples/frameworks/pytorch_setup/index.md).

* **Keep these references handy**:

		* The [Cheat Sheet](Cheatsheet.md) provides a quick reference for common
			commands and information about the Mila and DRAC clusters.

	* For a comprehensive reference of common terminal commands, see the
		[command line cheat sheet](https://cli-cheatsheet.readthedocs.io/).

!!! note
    Before running a minimal example, make sure to read
    [Running your code](Userguide_running_code.md), which explains how to submit
    jobs using Slurm and provides essential information about job submission
    arguments, partitions, and useful commands.
