---
title: Reproducibility and Sharing Results
description: >-
  Overview of practices for making cluster experiments reproducible and
  shareable — environment management, version control, dataset sharing, and
  research paper distribution.
---

# Reproducibility and Sharing Results

Reproducible research depends on being able to recreate an experiment's
environment, code, and data at any later point. This page gives a short
overview of the practices involved.

## Environment

Pin the exact versions of Python and its dependencies so an experiment can be
recreated on any node. See the [portability guide](python_uv.md) for managing
dependencies and lockfiles with `uv`.

## Version control

Track code changes with Git and push to a hosted remote such as GitHub or
GitLab. Tag or reference the exact commit used to produce a given set of
results, so the code behind any experiment or publication stays retrievable.

## Sharing datasets

To share a dataset with the Mila community, request its addition to
[`/network/datasets`](../technical_reference/clusters/mila/storage.md#datasets).
See the [Datasets](../technical_reference/general_theory/datasets.md) page for
ways to publicly share a Mila-hosted dataset — Academic Torrent, Google
Drive, and registering a DOI for citation.

For datasets that do not need to live on Mila's storage, the [Hugging Face
Hub](https://huggingface.co/datasets) hosts and versions datasets alongside
model cards and loading scripts, and GitHub (or GitLab) works well for small
datasets kept alongside the code that produces or consumes them.

## Research paper distribution

When publishing results, bundle the code, environment, and dataset references
readers need to reproduce them — for example through a public repository
release alongside the paper.
