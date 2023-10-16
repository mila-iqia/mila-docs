Cheat Sheet
***************

A "cheat sheet" is available to provide you with information about the Mila and DRAC clusters at a glance.

We printed a few hundred copies on cardboard paper in Fall 2023 and distributed them around Mila.
If you want to have a copy, you can always come to the IDT lab during office hours (usually Tuesday 3pm-5pm).

.. _cheatsheet-link: /_static/2023-09-07_IDT_cheatsheet.pdf

The :download:`IDT Cheat Sheet pdf <_static/2023-09-07_IDT_cheatsheet.pdf>`
is available if you want to access it online.
The layout of the pdf has been set to be compatible with the printers at Mila
so you can always print your own copy on regular paper
(hint: set printer scale 100% with no margins).

Keep in mind that the cheat sheet is not a replacement for the official documentation,
which is the original source of information.
Moreover, the official documentation is updated regularly, whereas the cheat sheet
is probably going to be updated once a year (around April when the new DRAC allocations are announced).

Comments and suggestions are welcome (`idt.cheatsheet@mila.quebec`).
Please also signal errors if you spot them before we do.


Errata
======

Here is a list of the known errors in the cheat sheet that will have to be fixed in the next version.

Partition preemption is not explained accurately on page 2
----------------------------------------------------------

The preemption on the Mila cluster is a bit more complicated than what is described in the cheat sheet.
The jobs from the `long` partition can be preempted to allow jobs in the `main` partition to run,
but jobs in `main` are never going to be preempted to allow for other jobs in `main`, no matter how much
of the "fair use" a user has already consumed.

This is why it can be considered rude (or just bad practice) to queue a large number of jobs
on `main` at midnight on the Mila cluster with a lot of free capacity.
On the next day, those jobs might still be running and much of the compute capacity
will be shared very unequally between the users, even though it seemed like a good strategy
to schedule jobs when the cluster was idle.

Default accounts on DRAC oversimplified
---------------------------------------

Technically, any professor can "sponsor" a user so that they have access to their "def-theirprofname" account.
It could be another professor besides the one supervising a given student,
even though it is not common practice (a notable exception being "def-bengioy").
A professor could even sponsor a student that is not affiliated with Mila in any way
for them to use the default account of that professor.

The cheat sheet implied that this was something that was more "automatic",
whereby every student has already access to the "def-theirprofname" account.


Minor typos
-----------

"Ask for things that are easy to schedule, the scheduler will be much nicer to you." -> Missing the word "and".

