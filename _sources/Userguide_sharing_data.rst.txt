Sharing Data with ACLs
======================

Regular permissions bits are extremely blunt tools: They control access through
only three sets of bits owning user, owning group and all others. Therefore,
access is either too narrow (``0700`` allows access only by oneself) or too wide
(``770`` gives all permissions to everyone in the same group, and ``777`` to
literally everyone).

ACLs (Access Control Lists) are an expansion of the permissions bits that allow
more fine-grained, granular control of accesses to a file. They can be used to
permit specific users access to files and folders even if conservative default
permissions would have denied them such access.


As an illustrative example, to use ACLs to allow ``$USER`` (**oneself**) to
share with ``$USER2`` (**another person**) a "playground" folder hierarchy in
Mila's scratch filesystem at a location

    ``$SCRATCH/X/Y/Z/...``

in a safe and secure fashion that allows both users to read, write, execute,
search and delete each others' files:

----


| **1.** Grant **oneself** permissions to access any **future** files/folders created
  by the other *(or oneself)*
| (``-d`` renders this permission a "default" / inheritable one)

.. code-block:: bash

   setfacl -Rdm user:${USER}:rwx  $SCRATCH/X/Y/Z/

----

.. note::
   The importance of doing this seemingly-redundant step first is that files
   and folders are **always** owned by only one person, almost always their
   creator (the UID will be the creator's, the GID typically as well). If that
   user is not yourself, you will not have access to those files unless the
   other person specifically gives them to you -- or these files inherited a
   default ACL allowing you full access.

   **This** is the inherited, default ACL serving that purpose.

| **2.** Grant **the other** permission to access any **future** files/folders created
  by the other *(or oneself)*
| (``-d`` renders this permission a "default" / inheritable one)

.. code-block:: bash

   setfacl -Rdm user:${USER2:?defineme}:rwx $SCRATCH/X/Y/Z/

----

| **3.** Grant **the other** permission to access any **existing** files/folders created
  by *oneself*.
| Such files and folders were created before the new default ACLs were added
  above and thus did not inherit them from their parent folder at the moment of
  their creation.

.. code-block:: bash

   setfacl -Rm  user:${USER2:?defineme}:rwx $SCRATCH/X/Y/Z/

----

| **4.** Grant **another** permission to search through one's hierarchy down to the
  shared location in question.

* **Non**-recursive (!!!!)
* May also grant ``:rx`` in unlikely event others listing your folders on the
  path is not troublesome or desirable.

.. code-block:: bash

   setfacl -m   user:${USER2:?defineme}:x   $SCRATCH/X/Y/
   setfacl -m   user:${USER2:?defineme}:x   $SCRATCH/X/
   setfacl -m   user:${USER2:?defineme}:x   $SCRATCH

.. note::
   The purpose of granting permissions first for *future* files and then for
   *existing* files is to prevent a **race condition** whereby after the first
   ``setfacl`` command the other person could create files to which the
   second ``setfacl`` command does not apply.

.. note::
   In order to access a file, all folders from the root (``/``) down to the
   parent folder in question must be searchable (``+x``) by the concerned user.
   This is already the case for all users for folders such as ``/``,
   ``/network`` and ``/network/scratch``, but users must explicitly grant access
   to some or all users either through base permissions or by adding ACLs, for
   at least ``/network/scratch/${USER:0:1}/$USER`` (= ``$SCRATCH``), ``$HOME`` and subfolders.

   To bluntly allow **all** users to search through a folder (**think twice!**),
   the following command can be used:

   .. code-block:: bash

      chmod a+X $SCRATCH

.. note::
  For more information on ``setfacl`` and path resolution/access checking,
  consider the following documentation viewing commands:

  * ``man setfacl``
  * ``man path_resolution``

Viewing and Verifying ACLs
--------------------------

.. code-block:: bash

   getfacl /path/to/folder/or/file
              1:  # file: somedir/
              2:  # owner: lisa
              3:  # group: staff
              4:  # flags: -s-
              5:  user::rwx
              6:  user:joe:rwx               #effective:r-x
              7:  group::rwx                 #effective:r-x
              8:  group:cool:r-x
              9:  mask::r-x
             10:  other::r-x
             11:  default:user::rwx
             12:  default:user:joe:rwx       #effective:r-x
             13:  default:group::r-x
             14:  default:mask::r-x
             15:  default:other::---

.. note::
  * ``man getfacl``
