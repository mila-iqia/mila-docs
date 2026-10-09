---
title: Install Git LFS on the Cluster
description: Install Git LFS on the cluster by downloading the binary 
    and adding it to the PATH.
---

# Git LFS

Git LFS (Large File Storage) extends Git to track large files without storing
them in the repository history. Git LFS is not preinstalled on the cluster,
but its binary can be downloaded and added to the `PATH`, after which the
`git lfs` commands become available in any repository.

## What this guide covers

* Get the Git LFS archive
* Extract the archive
* Add the `git-lfs` executable to the `PATH`
* Verify the installation

---

## Get the Git LFS archive

Git LFS is distributed as a tarball from the
[Git LFS releases page](https://github.com/git-lfs/git-lfs/releases/). Pick
the `git-lfs-linux-amd64-<VERSION>.tar.gz` asset for the desired version, and
download it on the cluster with `wget`:

```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v<VERSION>/git-lfs-linux-amd64-v<VERSION>.tar.gz
```

## Extract the archive

Extract the tarball with `tar`:

```bash
tar -xzf git-lfs-linux-amd64-v<VERSION>.tar.gz
```

!!! note
    `-x` extracts files, `-z` decompresses the gzip archive, and `-f`
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
