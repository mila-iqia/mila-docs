
Visual Studio Code
==================

One editor of choice for many researchers is VSCode. One feature of VSCode is remote editing through SSH. This allows you to edit files on the cluster as if they were local, open terminal sessions, and so on.

Making it work on the Mila cluster is a bit tricky. Here are the current best instructions as to how to make it all work.

TODO: Adapt Mattie's work from https://github.com/mila-iqia/mila-docs/issues/16, but this is somewhat involved, so ideally I think we should write a script to automate it all (e.g. a user could write ``mila-vscode dirname`` and it would get the allocation, find the compute node name, and do the connection).
