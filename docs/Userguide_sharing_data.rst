Sharing Data with ACLs
======================

As an illustrative example, to allow ``$USER`` to share with ``$USER2`` in
``rwx`` a hierarchy ``/network/scratch/$USER/X/Y/Z/...`` with ACLs:

----

| Grant **oneself** permissions to access any **future** files/folders created
  by the other *(or oneself)*
| (``-d`` renders this permission a "default" / inheritable one)

.. code-block:: bash

   setfacl -Rdm user:$USER:rwx  /network/scratch/$USER/X/Y/Z/

----

| Grant **another** permission to access any **future** files/folders created
  by the other *(or oneself)*
| (``-d`` renders this permission a "default" / inheritable one)

.. code-block:: bash

   setfacl -Rdm user:$USER2:rwx /network/scratch/$USER/X/Y/Z/

----

| Grant **another** permission to access any **existing** files/folders created
  by *oneself*.
| Such files and folders were created before the new default ACLs were added
  above and thus did not inherit them from their parent folder at the moment of
  their creation.

.. code-block:: bash

   setfacl -Rm  user:$USER2:rwx /network/scratch/$USER/X/Y/Z/

----

| Grant **another** permission to search through one's hierarchy down to the
  shared location in question.

* **Non**-recursive (!!!!)
* May also grant ``:rx`` in unlikely event others listing your folders on the
  path is not troublesome or desirable.

----

| In order to access a file, all folders from the root (``/``) down to the
  parent folder in question must be searchable (``+x``) by the concerned user.
  This is already the case for all users for folders such as ``/``, `/network`
  and ``/network/scratch``, but users must explicitly grant access to some or
  all users by adding ACLs for ``/network/scratch/$USER``, ``$HOME`` and
  subfolders.

.. code-block:: bash

   setfacl -m   user:$USER2:x   /network/scratch/$USER/X/Y/
   setfacl -m   user:$USER2:x   /network/scratch/$USER/X/
   setfacl -m   user:$USER2:x   /network/scratch/$USER/

----

| To bluntly allow **all** users to search through a folder (**think twice!**),
  the following command can be used:

.. code-block:: bash

        chmod a+x /network/scratch/$USER/

----

.. note::
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
