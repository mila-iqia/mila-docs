---
title: Install Git LFS on the Cluster
description: Install Git LFS on the cluster by downloading the Linux AMD64
  binary and adding it to the PATH.
---

# Install Git LFS on the Cluster

Git LFS (Large File Storage) extends Git to track large files without storing
them in the repository history. Git LFS is not preinstalled on the cluster,
but it runs on it: the Linux AMD64 binary can be downloaded, extracted, and
added to the `PATH`, after which the `git lfs` commands become available in
any repository.

## Before you begin

<div class="grid cards" markdown>

-   [:material-key:{ .lg .middle } __Logging in to the cluster__](../userguides/login.md)
    { .card }

    ---
    Connect to the cluster over SSH before installing a tool in a home
    directory.

&nbsp;

</div>

## What this guide covers

* Get the Git LFS Linux AMD64 archive, either directly on the cluster or on a
  local machine
* Extract the archive
* Add the `git-lfs` executable to the `PATH`
* Verify the installation

---

## Get the Git LFS archive

Git LFS is distributed as a tarball from the
[Git LFS releases page](https://github.com/git-lfs/git-lfs/releases/). Pick
the `git-lfs-linux-amd64-<VERSION>.tar.gz` asset for the desired version.

=== "Directly on the cluster"

    Download the archive on the cluster with `wget`:

    ```bash
    wget https://github.com/git-lfs/git-lfs/releases/download/v<VERSION>/git-lfs-linux-amd64-v<VERSION>.tar.gz
    ```

=== "On a local machine"

    Download the archive on a local machine, extract it, then upload the
    extracted `git-lfs-linux-amd64-<VERSION>/` directory to the cluster with
    `scp`:

    ```bash
    scp -r git-lfs-linux-amd64-<VERSION> mila:git-lfs-linux-amd64-<VERSION>
    ```

    When using this option, skip the extraction step below.

## Extract the archive

Extract the tarball with `tar`:

```bash
tar -xzf git-lfs-linux-amd64-v<VERSION>.tar.gz  # (1)!
```
{ .annotate }

1.  `-x` extracts files, `-z` decompresses the gzip archive, and `-f`
    specifies the archive file.

The command creates a `git-lfs-linux-amd64-<VERSION>/` directory containing
the `git-lfs` executable.

## Add git-lfs to the PATH

Add the directory containing the `git-lfs` executable to the `PATH`:

```bash
export PATH="$HOME/git-lfs-linux-amd64-<VERSION>:$PATH"
```

Append the same line to `~/.bashrc` to keep the setting across sessions:

```bash
echo 'export PATH="$HOME/git-lfs-linux-amd64-<VERSION>:$PATH"' >> ~/.bashrc
```

## Verify the installation

```bash
git lfs version
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
git-lfs/3.5.1 (GitHub; linux amd64; go 1.21.5)
```
</div>

A printed version number confirms Git LFS is installed and reachable on the
`PATH`.

---

## Key concepts

**Git LFS**
:   Git extension for versioning large files. LFS replaces large files in the
    repository history with text pointers, while storing the contents on a
    remote LFS server.
