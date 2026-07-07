---
title: Sharing data with ACLs
description: >-
  Set ACLs (Access Control Lists) to share data with other users on
  the cluster.
---

# Sharing data with ACLs

Regular permissions bits control access through only three sets of bits: owning
user, owning group, and all others. Access is therefore either too narrow (`700`
allows access only by the owner) or too wide (`770` grants all permissions to
everyone in the same group, and `777` to all users).

ACLs (Access Control Lists) expand on permissions bits to allow more
fine-grained access control. They permit specific users access to files and
folders even when conservative default permissions would otherwise deny it.

## Before you begin

<div class="grid cards" markdown>

-   [:material-key:{ .lg .middle } __Log in to the cluster__](login.md)
    { .card }

    ---
    Set up SSH access to the Mila cluster.

&nbsp;

</div>

!!! success "Requirements"
    - The cluster username of the collaborator to share data with.

## What this guide covers

* Grant yourself access to future files in the shared folder
* Grant a collaborator access to future files in the shared folder
* Grant a collaborator access to existing files in the shared folder
* Grant a collaborator search permissions on parent directories

---

## Setting ACLs

Use `setfacl` (set file access control list) to add ACLs to a file or
directory on the cluster.

The steps below show how to share `$SCRATCH/X/Y/Z/...` with a collaborator
in a way that allows both users to read, write, execute, search, and delete
each other's files.

```mermaid
flowchart LR
    A["Grant self —\nfuture files"] -->
    B["Grant collaborator —\nfuture files"] -->
    C["Grant collaborator —\nexisting files"] -->
    D["Grant collaborator —\nsearch on parents"]
```

### Granting yourself access to future files

Grant **yourself** permissions to access any **future** files/folders created by
the collaborator.

```bash
setfacl -Rdm user:${USER}:rwx $SCRATCH
```

!!! note
    Files and folders are almost **always** owned by their creator (the UID will
    be the creator's, the GID typically as well). If the creator is not
    yourself, those files will be inaccessible to your user unless the creator
    explicitly grants access to your user — or the files inherited a default ACL
    granting your user access.

    **This** is the inherited, default ACL serving that purpose.

    * The `-d` renders this permission a "default" / inheritable one.
    * The `-R` applies it recursively to all subfolders, so all files and
      folders created in the future within this hierarchy will inherit this ACL.
    * The `-m` modifies the ACL of the file or folder.

### Granting collaborators access to future files

Grant a collaborator permission to access any **future** files/folders created
within the shared hierarchy.

```bash
setfacl -Rdm user:${USER2:?collaborator-username}:rwx $SCRATCH/X/Y/Z/
```

### Granting collaborators access to existing files

Grant the collaborator permission to access any **existing** files/folders.
These were created before the new default ACLs were added and did not inherit
them at creation time.

```bash
setfacl -Rm user:${USER2:?collaborator-username}:rwx $SCRATCH/X/Y/Z/
```

!!! note
    Granting permissions for *future* files before *existing* files prevents a
    **race condition**: without this order, new files created between the two
    `setfacl` commands would not be covered by the second `setfacl` command.

### Granting search permissions on parent directories

Grant the collaborator permission to search through the folder hierarchy down to
the shared location. This step is non-recursive and must be run for each folder
on the path to the shared location.

```bash
setfacl -m user:${USER2:?collaborator-username}:x $SCRATCH/X/Y/
setfacl -m user:${USER2:?collaborator-username}:x $SCRATCH/X/
setfacl -m user:${USER2:?collaborator-username}:x $SCRATCH
```

!!! tip
    Also grant `:rx` to allow the collaborator to list the parent folders as
    well.

!!! warning
    To access a file, all folders from the root (`/`) down to the parent folder
    must be searchable (`+x`) by the collaborator. This is already the case for
    `/`, `/network`, and `/network/scratch`. At least
    `/network/scratch/${USER:0:1}/$USER` (= `$SCRATCH`), `$HOME`, and subfolders
    require explicit grants — via base permissions or ACLs.

!!! note
    For more information on `setfacl` and path resolution/access checking, see:

    * `man setfacl`
    * `man path_resolution`

### Quick reference

| Goal | Command | Note |
|------|---------|------|
| Grant calling user access to future files | `setfacl -Rdm user:$USER:rwx <shared-dir>` | Inheritable default ACL |
| Grant collaborator access to future files | `setfacl -Rdm user:${USER2:?collaborator-username}:rwx <shared-dir>` | |
| Grant collaborator access to existing files | `setfacl -Rm user:${USER2:?collaborator-username}:rwx <shared-dir>` | No `-d` flag |
| Grant search on a parent directory | `setfacl -m user:${USER2:?collaborator-username}:x <parent-dir>` | One per dir level |

## Removing ACLs

Use the `-x` option of `setfacl` to remove the ACL entry for a specific user.

```bash
setfacl -x user:${USER2:?collaborator-username} $SCRATCH/X/Y/Z/
```

To remove all ACL entries for a file or folder, use the `-b` option.

```bash
setfacl -b $SCRATCH/X/Y/Z/
```

## Viewing and verifying ACLs

Use `getfacl` (get file access control list) to display the ACLs of a file
or directory.

```bash
getfacl /path/to/folder/or/file
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
# file: somedir/
# owner: lisa
# group: staff
# flags: -s-
user::rwx
user:joe:rwx               #effective:r-x
group::rwx                 #effective:r-x
group:cool:r-x
mask::r-x
other::r-x
default:user::rwx
default:user:joe:rwx       #effective:r-x
default:group::r-x
default:mask::r-x
default:other::---
```
</div>

!!! note
    For more information on `getfacl`, see:

    * `man getfacl`

---

## Key concepts

**ACL (Access Control List)**
:   An extension of POSIX permissions that assigns read, write, and execute
    permissions to specific users or groups beyond the standard
    owner/group/other model.

**Default ACL**
:   An inheritable ACL entry (set with `-d`) that new files and directories
    automatically inherit from their parent folder at creation time.

`setfacl`
:   Command-line tool for setting file access control lists on Linux
    filesystems.

`getfacl`
:   Command-line tool for displaying file access control lists on Linux
    filesystems.

`$SCRATCH`
:   Environment variable pointing to the user's personal directory on Mila's
    scratch filesystem (`/network/scratch/${USER:0:1}/$USER`).
