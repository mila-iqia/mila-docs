---
title: Development
description: >-
  Tools to write, run, and debug code interactively on the clusters.
---

# Development

Research code usually needs to be written, tested, and debugged before it can
run as a batch job. These tools support that day-to-day development work on the
clusters, from interactive notebooks to full-featured editors and command-line
helpers.

!!! warning
    Do not run editors, notebooks, or other heavy processes on the login
    nodes. Each of these tools first allocates a compute node through Slurm,
    then connects to it.

<div class="grid cards" markdown>

-   [:simple-jupyter:{ .lg .middle } __JupyterHub__](jupyterhub.md)
    { .card }

    ---
    Start a JupyterLab session on a compute node from a web browser, with no
    SSH setup required.

-   [:material-microsoft-visual-studio-code:{ .lg .middle } __VSCode__](VSCode.md)
    { .card }

    ---
    Edit, run, and debug code on a compute node from a local VSCode window
    through remote SSH.

-   [:material-console-line:{ .lg .middle } __milatools__](https://github.com/mila-iqia/milatools)
    { .card }

    ---
    A command-line tool that sets up SSH access to the cluster and opens
    VSCode on a compute node with a single command.

</div>
