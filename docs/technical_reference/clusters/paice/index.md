---
title: PAICE Clusters
description: Access and use the Pan-Canadian AI Compute Environment (PAICE)
  clusters reserved for CIFAR AI Chairs.
---

# Pan-Canadian AI Compute Environment Clusters

The PAICE (Pan-Canadian AI Compute Environment) clusters are part of the
[Digital Research Alliance of Canada](https://alliancecan.ca/) (DRAC)
infrastructure, dedicated exclusively to AI research and reserved for
[CIFAR AI Chairs](https://cifar.ca/ai/). Unlike general DRAC clusters,
PAICE clusters use AIP (Artificial Intelligence Project) allocations and
follow a tiered access model.

!!! note
    When publishing research using the Pan-Canadian AI Compute Environment
    (PAICE), acknowledge the Digital Research Alliance of Canada, your
    specific regional partner and the AI institute that manages your
    cluster.

## Account creation

PAICE accounts are hosted on the DRAC [CCDB](https://ccdb.alliancecan.ca) portal.
Account creation follows the same steps as for other DRAC clusters. See the
[DRAC Clusters guide](../drac/index.md#account-creation) for the full
process.

As a Mila researcher, you can request access of all the resources in the
*Artificial Intelligence* tabs in *Resources > Access Systems* at CCDB.

### Access to AIP allocation

Before you will be able to submit jobs on PAICE clusters, your professor
must add you to their AIP (Artificial Intelligence Project) allocation.
To be added, share **your** CCRI.

### Connect to the clusters

Connecting to PAICE clusters follows the same steps as DRAC clusters,
including setting up SSH keys and configuring multifactor authentication
on the CCDB portal. See the
[DRAC Clusters guide](../drac/index.md#connect-to-the-clusters) for
instructions.

### Renewal

Account renewal follows the same annual process as other DRAC clusters.
See the [DRAC Clusters guide](../drac/index.md#renewal) for instructions.

## Clusters

These clusters use AIP allocations (``--account=aip-${PI_NAME}``), where
``${PI_NAME}`` is the name of your supervising professor. Regular DRAC
allocations and the Mila global allocation will not work.

Access priority is distributed across three tiers based on supervisor
affiliation and researcher location. By default, tier 1 and tier 2
together receive 85% of each cluster's resources, while tier 3 receives
10%. Contact your supervisor to determine which tier applies to your
research group.

The table below provides information on the allocation depending of the default
tier of a Mila researcher for the period which spans from April 7, 2026 to Spring 2027.

| Cluster                             | CPUs | RGUs allocated | # GPU equiv | Model               | Unrestricted internet |
| ----------------------              | ---- | -------------- | ----------- | --------            | -----                 |
| [TamIA](#tamia) <br> tier1 + tier2  | 435  | 1738           | 143         | H100-80G <br> H200  | No                    |
| [Killarney](#killarney) <br> tier3  | 0    | 794            | 75          | H100-80G <br> L40S  | Yes                   |
| [Vulcan](#vulcan) <br> tier3        | 0    | 850            | 82          | L40S                | No                    |

Check the current status of the clusters on the [DRAC status page](https://status.alliancecan.ca/).

### TamIA

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/TamIA/en)

Cluster managed by Mila and [Calcul Québec](https://www.calculquebec.ca/),
located at [Université Laval](https://www.ulaval.ca/).
Compute resources in TamIA are not assigned to jobs on a per-CPU, but on
a per-node basis. No internet access on compute nodes.

### Killarney

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/Killarney/en)

Cluster managed by [Vector](https://vectorinstitute.ai/) and
[SciNet](https://scinethpc.ca/), located at the
[University of Toronto](https://www.utoronto.ca/).

### Vulcan

[Digital Research Alliance of Canada doc](https://docs.alliancecan.ca/wiki/Vulcan/en)

Cluster managed by the
[University of Alberta](https://www.ualberta.ca/en/index.html) and
[AMII](https://www.amii.ca/), located at the University of Alberta.
No internet access on compute nodes.

## Launching jobs

Users must specify the AIP allocation using the flag
``--account=aip-${PI_NAME}``, where ``${PI_NAME}`` is the name of the
supervising professor. To launch a CPU-only job:

```bash
sbatch --time=1:00:00 --account=aip-${PI_NAME} job.sh
```

To launch a GPU job:

```bash
sbatch --time=1:00:00 --account=aip-${PI_NAME} --gres=gpu:1 job.sh
```

To get an interactive session:

```bash
salloc --time=1:00:00 --account=aip-${PI_NAME} --gres=gpu:1
```

!!! note "TamIA: per-node allocation"
    On TamIA, compute resources are allocated on a per-node basis, not
    per-CPU. Refer to the
    [TamIA documentation](https://docs.alliancecan.ca/wiki/TamIA/en)
    for node specifications and submission guidelines.

The full documentation for job launching on Alliance clusters can be
found [here](https://docs.alliancecan.ca/wiki/Running_jobs#).

## Storage

| Storage          | Path                      | Usage                                                         |
| ---------------- | ------------------------- | ------------------------------------------------------------- |
| `$HOME`          | `/home/<user>/`           | Code, specific libraries                                      |
| `$HOME/projects` | `/project/<project>`      | Compressed raw datasets                                       |
| `$SCRATCH`       | `/scratch/<user>`         | Processed datasets, experimental results, logs of experiments |
| `$SLURM_TMPDIR`  | (on compute node)         | Temporary job data or results                                 |

When a series of experiments is finished, results should be transferred
back to Mila servers.

More details on storage can be found on the [DRAC Clusters guide](../drac/index.md#storage) or
on [DRAC wiki](https://docs.alliancecan.ca/wiki/Storage_and_file_management).

## Using CometML and Wandb

Some compute nodes don't have access to the internet, but there is a special
module that can be loaded in order to allow training scripts to access some 
specific servers, which includes the necessary servers for using CometML and
Wandb ("Weights and Biases").

```bash
module load httpproxy
```

More documentation about this can be found [here](https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb)).

!!! note

    Be careful when using Wandb with `httpproxy`. It does not support sending
    artifacts and wandb's logger will hang in the background when your training
    is completed, wasting resources until the job times out. It is recommended
    to use the offline mode with wandb instead to avoid such waste.