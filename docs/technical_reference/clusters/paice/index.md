# Pan-Canadian AI Compute Environment Clusters

TODO : intro
The PAICE (Pan-Canadian AI Compute Environment) clusters are also part of DRAC, but are especially develop for CIFAR AI Chairs
and are dedicated to artificial intelligence.

!!! note
    When publishing research using the Pan-Canadian AI Compute Environment (PAICE), acknowledge the Digital Research Alliance of Canada,
    your specific regional partner and the AI institute that manages your specific cluster.

## Account Creation

TODO : follow DRAC Clusters steps

As a Mila researcher, you can request access of all the resources in the *HPC* and *Artificial Intelligence* tabs in *Resources > Access Systems* at CCDB.

### Connect to the clusters

TODO : follow DRAC Clusters steps

### Renewal

TODO : follow DRAC Clusters steps

## Clusters

Theses clusters use AIP allocations (``--account=aip-${PI_NAME}``), where ``${PI_NAME}`` is the name of
your supervising professor. Your professor must add you to their AIP allocation before
you will be able to submit jobs. Regular DRAC allocations and the Mila global allocation will not work.

The table below provides information on the allocation depending of the tier of the professor
(by default 85% of the resources of the cluster for tier1+tier2 and 10% for tier3)
for the period which spans from April 7, 2026 to Spring 2027.

| Cluster                             | CPUs | RGUs allocated | # GPU equiv | Model               | Unrestricted internet |
| ----------------------              | ---- | -------------- | ----------- | --------            | -----                 |
| [TamIA](#TamIA) <br> tier1 + tier2  | 435  | 1738           | 143         | H100-80G <br> H200  | No                    |
| [Killarney](#killarney) <br> tier3  | 0    | 794            | 75          | H100-80G <br> L40S  | Yes                   |
| [Vulcan](#vulcan) <br> tier3        | 0    | 850            | 82          | L40S                | No                    |


### TamIA

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/TamIA/en)

Cluster managed by Mila and [Calcul Québec](https://www.calculquebec.ca/), located at [Université Laval](https://www.ulaval.ca/). 
Compute resources in TamIA are not assigned to jobs on a per-CPU, but on a per-node basis. No internet access on compute nodes.

### Killarney

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/Killarney/en)

Cluster managed by [Vector](https://vectorinstitute.ai/) and [SciNet](https://scinethpc.ca/), located at the [University of Toronto](https://www.utoronto.ca/).

### Vulcan

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/Vulcan/en)

Cluster managed by the [University of Alberta](https://www.ualberta.ca/en/index.html) and [AMII](https://www.amii.ca/), located at the University of Alberta. No internet access on compute nodes.