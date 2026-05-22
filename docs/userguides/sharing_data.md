---
title: Sharing data with ACLs
description: >-
  Set ACLs (Access Control Lists) to share data with other users on the cluster.
---

# Sharing data with ACLs

Regular permissions bits are extremely blunt tools. They control access through
only three sets of bits owning user, owning group and all others. Therefore,
access is either too narrow (`700` allows access only by oneself) or too wide
(`770` gives all permissions to everyone in the same group, and `777` to
literally everyone).

ACLs (Access Control Lists) are an expansion of the permissions bits that allow
more fine-grained control of accesses to a file. They can be used to
permit specific users access to files and folders even if conservative default
permissions would have denied them such access.

## Setting ACLs

Use `setfacl` (set file access control list) to add ACLs to a file or directory on the cluster.

The following example shows how to use ACLs to allow `$USER` (**you**) to
share `$SCRATCH/X/Y/Z/...` (**a folder hierarchy in Mila's scratch filesystem**)
with `$USER2` (**another person**) in a safe and secure fashion that allows both users to read, write, execute,
search and delete each other's files.

### Granting yourself access to future files

Grant **yourself** permissions to access any **future** files/folders created by the other *(or oneself)*.

```bash
setfacl -Rdm user:${USER}:rwx $SCRATCH/X/Y/Z/
```

!!! note
    The importance of doing this seemingly-redundant step first is that files
    and folders are **always** owned by only one person, almost always their
    creator (the UID will be the creator's, the GID typically as well). If that
    user is not yourself, you will not have access to those files unless the
    other person specifically gives them to you -- or these files inherited a
    default ACL allowing you full access.

    **This** is the inherited, default ACL serving that purpose.

    * The `-d` renders this permission a "default" / inheritable one. 
    * The `-R` applies it recursively to all subfolders, so that all files and
    folders created in the future within this hierarchy will inherit this ACL. 
    * The `-m` modifies the ACL of the file or folder.

### Granting other users access to future files

Grant **the other** permission to access any **future** files/folders created
by the other *(or oneself)*.

```bash
setfacl -Rdm user:${USER2:?defineme}:rwx $SCRATCH/X/Y/Z/
```

### Granting other users access to existing files

Grant **the other** permission to access any **existing** files/folders created
by *oneself*.
Such files and folders were created before the new default ACLs were added
above and thus did not inherit them from their parent folder at the moment of
their creation.

```bash
setfacl -Rm  user:${USER2:?defineme}:rwx $SCRATCH/X/Y/Z/
```

!!! note
    The purpose of granting permissions first for *future* files and then for
    *existing* files is to prevent a **race condition** whereby after the first
    ``setfacl`` command the other person could create files to which the
    second ``setfacl`` command does not apply.

### Granting search permissions to access the shared location

Grant **another** permission to search through one's hierarchy down to the
shared location in question. This step is non-recursive and must be run for each folder
on the path to the shared location.

```bash
setfacl -m   user:${USER2:?defineme}:x   $SCRATCH/X/Y/
setfacl -m   user:${USER2:?defineme}:x   $SCRATCH/X/
setfacl -m   user:${USER2:?defineme}:x   $SCRATCH
```

!!! tip
    Also grant `:rx` if allowing the other user to list the parent folders is acceptable.

!!! warning
    In order to access a file, all folders from the root (``/``) down to the
    parent folder in question must be searchable (``+x``) by the concerned user.
    This is already the case for all users for folders such as ``/``,
    ``/network`` and ``/network/scratch``, but users must explicitly grant access
    to some or all users either through base permissions or by adding ACLs, for
    at least ``/network/scratch/${USER:0:1}/$USER`` (= ``$SCRATCH``), ``$HOME`` and subfolders.

!!! note
    For more information on `setfacl` and path resolution/access checking,
    consider the following documentation viewing commands:
    
    * `man setfacl`
    * `man path_resolution`

## Removing ACLs
To remove access for a user, use the `-x` option of `setfacl` to remove the ACL entry for that user.

```bash
setfacl -x user:${USER2:?defineme} $SCRATCH/X/Y/Z/
```

To remove all ACL entries for a file or folder, use the `-b` option of `setfacl`.

```bash
setfacl -b $SCRATCH/X/Y/Z/
```

## Viewing and verifying ACLs
Use `getfacl` (get file access control list) to display the ACLs of a file or directory.

```bash
getfacl /path/to/folder/or/file
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

!!! note
    For more information on `getfacl` consider the following documentation viewing command:

    * `man getfacl`