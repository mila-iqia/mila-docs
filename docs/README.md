---
title: Mila Technical Documentation
description: Technical documentation for Mila's computing infrastructure.
---

# Mila technical documentation


Welcome to Mila's technical documentation. If this is your first time here, we
recommend you start by checking out the [short quick start guide](Userguide_quick_start.md).

<!-- include: Acknowledgement_text.md -->
{%
    include-markdown "Acknowledgement_text.md"
%}

## Support
To reach the Mila infrastructure support, please [submit a support ticket.](https://mila-iqia.atlassian.net/servicedesk/customer/portals)


## Contribution
If you find any errors in the documentation, missing or unclear sections, or would simply like to contribute, please open an issue or make a pull request on the [github page](https://github.com/mila-iqia/mila-docs).

## Documentation

<!--nav-->
- Introduction
    - [Purpose of this documentation](Purpose.md)
    - [Contributing](CONTRIBUTING.md)
- How-tos and Guides
    - [Quick Start](Userguide_quick_start.md)
    - [Logging in to the cluster](Userguide_login.md)
    - [Running your code](Userguide_running_code.md)
    - [Portability concerns and solutions](Userguide_portability.md)
    - [Using containers](Userguide_containers.md)
    - [Sharing Data with ACLs](Userguide_sharing_data.md)
    - [Contributing datasets](Userguide_datasets.md)
    - [Data Transmission using Globus Connect Personal](Userguide_data_transfer.md)
    - [Advanced SLURM usage and Multiple GPU jobs](Userguide_multigpu.md)
    - [Multiple Nodes](Userguide_multinode.md)
    - [Weights and Biases (WandB)](Userguide_wandb.md)
    - [Comet](Userguide_comet.md)
    - [JupyterHub](Userguide_jupyterhub.md)
    - [Singularity](Userguide_singularity.md)
    - [Frequently asked questions (FAQ)](Userguide_faq.md)
- Systems and services
    - [Computing infrastructure and policies](Information.md)
        - [Roles and computing resources](Information_roles_and_resources.md)
        - [Node profile description](Information_nodes.md)
        - [Storage](Information_storage.md)
        - [Data sharing policies](Information_sharing_policies.md)
        - [Data Transmission](Information_data_transmission.md)
        - [Monitoring](Information_monitoring.md)
    - [Computational resources outside of Mila](Extra_compute.md)
- Minimal Examples
    - [Software Frameworks](examples/frameworks/index.md)
        - [PyTorch Setup](examples/frameworks/pytorch_setup/index.md)
        - [Jax Setup](examples/frameworks/jax_setup/index.md)
        - [Jax](examples/frameworks/jax/index.md)
    - [Distributed Training](examples/distributed/index.md)
        - [Single GPU Job](examples/distributed/single_gpu/index.md)
        - [Multi-GPU Job](examples/distributed/multi_gpu/index.md)
        - [Multi-node Job](examples/distributed/multi_node/index.md)
    - [Good Practices](examples/good_practices/index.md)
        - [Checkpointing](examples/good_practices/checkpointing/index.md)
        - [Weights & Biases (wandb) setup](examples/good_practices/wandb_setup/index.md)
        - [Launch many jobs from same shell script](examples/good_practices/launch_many_jobs/index.md)
        - [Hyperparameter Optimization with Orion](examples/good_practices/hpo_with_orion/index.md)
        - [Launch many tasks on the same GPU](examples/good_practices/many_tasks_per_gpu/index.md)
        - [Launch many jobs using SLURM job arrays](examples/good_practices/slurm_job_arrays/index.md)
    - [Advanced Examples](examples/advanced/index.md)
        - [Multi-Node / Multi-GPU ImageNet Training](examples/advanced/imagenet/index.md)
    - [🔗 Research Project Template](https://mila-iqia.github.io/ResearchTemplate)
- General Theory
    - [What is a computer cluster?](Theory_cluster_parts.md)
    - [Unix](Theory_cluster_unix.md)
    - [The workload manager](Theory_cluster_batch_scheduling.md)
    - [Processing data](Theory_cluster_data.md)
    - [Software on the cluster](Theory_cluster_software_deps.md)
- Extras
    - [Acknowledging Mila](Acknowledgement.md)
    - [Mila Datasets](https://datasets.server.mila.quebec/)
    - [Audio and video resources at Mila](Audio_video.md)
    - [Visual Studio Code](VSCode.md)
    - [Who, what, where is IDT](IDT.md)
    - [Cheat Sheet](Cheatsheet.md)
    - [Environmental Impact](Environmental_impact.md)


