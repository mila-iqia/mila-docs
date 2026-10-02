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

-   [:material-docker:{ .lg .middle } __Podman__](podman.md)
    { .card }

    ---
    Recommended on the Mila cluster.

-   [:material-package-variant-closed:{ .lg .middle } __Singularity__](singularity.md)
    { .card }

    ---
    Available on the Mila and DRAC clusters.

</div>
