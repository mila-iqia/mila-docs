---
title: Mila Technical Documentation
description: Technical documentation for Mila's computing infrastructure.
---

# Mila technical documentation

Welcome to Mila's technical documentation. If this is your first time here, we
recommend you start by checking out the [short quick start guide](getting_started/).


<div class="grid cards" markdown>



-   <a href="getting_started" class="full-card">
    :material-clock-fast:{ .lg .middle } __Getting started__

    ---

    Learn to log in to the cluster, run your first job, and train your first mode

    </a>


-   <a href="userguides" class="full-card">
    :material-map:{ .lg .middle } __How-tos and Guides__

    ---

    Discover more advanced guides to help you in your research

    </a>

-   <a href="ai" class="full-card">
    :material-robot:{ .lg .middle } __AI agents__

    ---

    Coming soon!

    Powercharge your AI agent with curated resources

    </a>

-   <a href="technical_reference" class="full-card">
    :material-book-open-page-variant:{ .lg .middle } __Technical reference__

    ---

    Find advanced notions to better understand how everything works

    </a>

-   <a href="toolbox" class="full-card">
    :material-hammer-wrench:{ .lg .middle } __Toolbox__

    ---

    Obtain tools for your projects

    </a>

-   <a href="help" class="full-card">
    :material-account-question:{ .lg .middle } __Get help__

    ---

    Find how to solve your problems

    :octicons-arrow-right-24: Find your answers in the FAQ

    :octicons-arrow-right-24: Join us on Slack (#mila-cluster)

    :octicons-arrow-right-24: Ask your question to IT support
    
    :octicons-arrow-right-24: Join us at the Tue 3-5pm/Wed 2-4pm Office Hours in Lab A

    </a>

</div>


{% include-markdown "home/purpose.md" %}


## Contribution
If you find any errors in the documentation, missing or unclear sections, or would simply like to contribute, please open an issue or make a pull request on the [github page](https://github.com/mila-iqia/mila-docs).

## Acknowledging Mila

{% include-markdown "home/acknowledgement.md" %}


## Documentation

<!--nav-->
- [Introduction](README.md)
    - [Cheat Sheet](home/cheatsheet.md)
    - [Acknowledging Mila](home/acknowledgement.md)
    - [About us](home/teams.md)
    - [🔗 Mila intranet](https://intranet.mila.quebec/)
- [Getting started](getting_started/index.md)
    - [Run Your First Job](getting_started/my_first_job.md)
    - [Train Your First Model](getting_started/train_first_model.md)
- [How-tos and Guides](userguides/index.md)
    - [Multi-Factor Authentication (MFA) for Cluster Access](Userguide_login_mfa.md)
    - [Logging in to the cluster](userguides/login.md)
    - [Running your code](userguides/running_code.md)
    - [Sharing Data with ACLs](userguides/sharing_data.md)
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
- [AI agents](ai/index.md)
- [Technical reference](technical_reference/index.md)
    - [Cheatsheet](technical_reference/cheatsheet.md)
    - [Glossary](technical_reference/glossary.md)
    - Clusters
        - [Mila cluster](technical_reference/clusters/mila/index.md)
            - [Roles and computing resources](technical_reference/clusters/mila/roles_and_resources.md)
            - [Node profile description](technical_reference/clusters/mila/nodes.md)
            - [Storage](technical_reference/clusters/mila/storage.md)
            - [Data sharing policies](technical_reference/clusters/mila/sharing_policies.md)
            - [Data Transmission](technical_reference/clusters/mila/data_transmission.md)
            - [Monitoring](technical_reference/clusters/mila/monitoring.md)
            - [Environmental Impact](technical_reference/clusters/mila/environmental_impact.md)
            - [🔗 Mila intranet](https://sites.google.com/mila.quebec/mila-intranet)
        - [External clusters](technical_reference/clusters/external/index.md)
            - [DRAC clusters](technical_reference/clusters/drac/index.md)
    - General Theory
        - [What is a computer cluster?](technical_reference/general_theory/cluster_parts.md)
        - [Unix](technical_reference/general_theory/unix.md)
        - [The workload manager](technical_reference/general_theory/batch_scheduling.md)
        - [Processing data](technical_reference/general_theory/data.md)
        - [Software on the cluster](technical_reference/general_theory/software_deps.md)
        - [Portability concerns and solutions](technical_reference/general_theory/portability.md)
        - [Using containers](technical_reference/general_theory/containers.md)
        - [Contributing datasets](technical_reference/general_theory/datasets.md)
        - [Data Transmission using Globus Connect Personal](technical_reference/general_theory/data_transfer.md)
        - [Advanced SLURM usage and Multiple GPU jobs](technical_reference/general_theory/multigpu.md)
        - [Multiple Nodes](technical_reference/general_theory/multinode.md)
        - [Useful links](technical_reference/useful_links.md)
- [Toolbox](toolbox/index.md)
    - [Comet](toolbox/comet.md)
    - [JupyterHub](toolbox/jupyterhub.md)
    - [Singularity](toolbox/singularity.md)
    - [VSCode](toolbox/VSCode.md)
    - [Weights and Biases (WandB)](toolbox/wandb.md)
    - [🔗 Research Project Template](https://mila-iqia.github.io/ResearchTemplate)
- [Get help](help/index.md)
    - [Frequently asked questions (FAQ)](help/faq.md)

