---
title: Roles and computing resources
description: How Mila affiliation status determines cluster access, and an overview of the Mila cluster's computing resources.
---

# Roles and computing resources

Mila assigns one of two affiliation statuses to its professors and principal
investigators:

1. **Core researchers**
2. **Affiliated researchers**

Students, postdocs, and other trainees inherit their access rights from their
supervisor's status. If your supervisor (or co-supervisor) is a core researcher,
access to the Mila computing cluster is granted. If both your supervisor and
co-supervisor are affiliated researchers, access is not granted.

To determine your own status, check the Mila affiliation of your supervisor and,
if applicable, your co-supervisor.

## Overview of available computing resources at Mila

The Mila cluster is a heterogeneous cluster with a variety of node types
(see [Node profile description](nodes.md)) and is to be used for regular development.
It uses [Slurm](../../general_theory/slurm.md) to schedule jobs.

### Mila cluster versus DRAC / PAICE clusters

There are a lot of commonalities between the Mila cluster and the clusters from
[Digital Research Alliance of Canada (DRAC / the Alliance)](../drac/index.md)
and [Pan-Canadian AI Compute Environment (PAICE)](../paice/index.md) (job
scheduling with Slurm, filesystem, etc). At the time being, core researchers
also have access to a large allocation of resources on the Alliance clusters
through Mila's global allocation.

The main distinguishing factor is that we have more control over our own cluster
than we have over the external ones. Notably, the compute nodes in the Mila
cluster all have unrestricted access to the Internet, which is not the case in
general (although some clusters allow it).

Mila students are advised to use a healthy diet of a mix of Mila and
DRAC / PAICE clusters. This is especially true when a preferred cluster
is oversubscribed and switching to a different cluster is possible.

See the [clusters](../index.md) page for a list of all available clusters.

### Guarantees about one GPU as absolute minimum

There are certain guarantees that the Mila cluster tries to honor when it comes
to giving *at minimum* one GPU per student, all the time, to be used in
interactive mode. This is strictly better than "one GPU per student on average"
because it's a floor meaning that, at any time, a GPU can be requested and
obtained immediately (although processing the request through Slurm might
take a minute).

Interactive sessions are also possible on the Alliance clusters, and there are
generally special rules that provide quicker access to resources when
requested for a very short duration (for testing code before queueing long
jobs). The same guarantee does not apply on the Mila clusters, however.
