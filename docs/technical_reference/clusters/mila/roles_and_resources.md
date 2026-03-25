# Roles and authorizations


There are mainly two types of researchers statuses at Mila :

1. Core researchers
2. Affiliated researchers

This is determined by Mila policy. Core researchers have access to the Mila
computing cluster. See your supervisor's Mila status to know what is your own
status.


## Overview of available computing resources at Mila


The Mila cluster is to be used for regular development and relatively small
number of jobs (< 5). It is a heterogeneous cluster. It uses
[SLURM](Userguide_running_code.md) to schedule jobs.


### Mila cluster versus Digital Research Alliance of Canada clusters


There are a lot of commonalities between the Mila cluster and the clusters from
Digital Research Alliance of Canada (the Alliance). At the time being, the
Alliance clusters where we have a large allocation of resources are `beluga`,
`cedar`, `graham` and `narval`. We also have comparable computational resources
in the Mila cluster, with more to come.

The main distinguishing factor is that we have more control over our own cluster
than we have over the ones at the Alliance. Notably, also, the compute nodes in
the Mila cluster all have unrestricted access to the Internet, which is not the
case in general for the Alliance clusters (although `cedar` does allow it).

At the current time of this writing (June 2021), Mila students are advised to
use a healthy diet of a mix of Mila and Alliance clusters. This is especially
true in times when your favorite cluster is oversubscribed, because you can
easily switch over to a different one if you are used to it.


### Guarantees about one GPU as absolute minimum


There are certain guarantees that the Mila cluster tries to honor when it comes
to giving *at minimum* one GPU per student, all the time, to be used in
interactive mode. This is strictly better than "one GPU per student on average"
because it's a floor meaning that, at any time, you should be able to ask for
your GPU, right now, and get it (although it might take a minute for the request
to be processed by SLURM).

Interactive sessions are possible on the Alliance clusters, and there are
generally special rules that allow you to get resources more easily if you
request them for a very short duration (for testing code before queueing long
jobs). You do not get the same guarantee as on the Mila cluster, however.
