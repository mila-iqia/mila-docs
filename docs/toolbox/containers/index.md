---
title: Containers
description: >-
  What containers are and which container tools work on the clusters.
---

# Containers

A container packages an application with everything it needs to run (system
libraries, packages, configuration...), so it behaves the same on a
laptop and on any cluster. A container is built from an **image**, a
read-only template usually pulled from a registry such as
[Docker Hub](https://hub.docker.com/).

!!! note
    For most Python-only projects, a virtual environment is simpler. Use a
    container when the software needs system packages that are not available on
    the cluster, or when the exact same environment must run everywhere.

Docker is not available on the clusters, but Docker images work with these
tools:

<div class="grid cards" markdown>

-   [:simple-podman:{ .lg .middle } __Podman__](podman.md)
    { .card }

    ---
    A drop-in replacement for Docker that runs containers without root
    privileges, using the same commands.

-   [:material-package-variant-closed:{ .lg .middle } __Singularity__](singularity.md)
    { .card }

    ---
    A container tool designed for shared HPC clusters, which runs each
    container as a single image file.

</div>
