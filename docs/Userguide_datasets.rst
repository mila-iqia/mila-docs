Contributing datasets
=====================


If a dataset could help the research of others at Mila, `this form
<https://forms.gle/vDVwD2rZBmYHENgZA>`_ can be filled to request its addition
to `/network/datasets <Information.html#storage>`_.

Those datasets can be mirrored to the Béluga cluster in
``~/projects/rrg-bengioy-ad/data/curated/`` if they follow Compute Canada's
`good practices on data
<https://docs.computecanada.ca/wiki/AI_and_Machine_Learning#Managing_your_datasets>`_.


Publicly share a Mila dataset
-----------------------------

Mila offers two ways to publicly share a Mila dataset:

* `Academic Torrent <https://academictorrents.com>`_
* `Google Drive
  <https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_

Note that these options are not mutually exclusive and both can be used.


Academic Torrent
^^^^^^^^^^^^^^^^

Mila hosts/seeds some datasets created by the Mila community through `Academic
Torrent <https://academictorrents.com>`_. The first step is to create `an
account and a torrent file <https://academictorrents.com/upload.php>`_.

Then drop the dataset in ``/network/scratch/.transit_datasets`` and send the
Academic Torrent URL to `Mila's helpdesk <https://it-support.mila.quebec>`_. If
the dataset does not reside on the Mila cluster, only the Academic Torrent URL
would be needed to proceed with the initial download. Then you can delete /
stop sharing your copy.

.. note::
   * Avoid mentioning *dataset* in the name of the dataset
   * Avoid capital letters, special charaters (including spaces) in files and
     directories names. Spaces can be replaced by hyphens (``-``).
   * Multiple archives can be provided to spread the data (e.g. dataset splits,
     raw data, extra data, ...)


Google Drive
^^^^^^^^^^^^

Only a member of the staff team can upload to `Mila's Google Drive
<https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_
which requires to first drop the dataset in
``/network/scratch/.transit_datasets``. Then, contact `Mila's helpdesk
<https://it-support.mila.quebec>`_ and provide the following informations:

* directory containing the archived dataset (zip is favored) in
  ``/network/scratch/.transit_datasets``
* the name of the dataset
* a licence in ``.txt`` format. One of the `the creative common
  <https://creativecommons.org/about/cclicenses/>`_ licenses can be used. It is
  recommended to at least have the *Attribution* option. The *No Derivatives*
  option is discouraged unless the dataset should not be modified by others.
* MD5 checksum of the archive
* the arXiv and GitHub URLs (those can be sent later if the article is still in
  the submission process)
* instructions to know if the dataset needs to be ``unzip``\ed, ``untar``\ed or
  else before uploading to Google Drive

.. note::
   * Avoid mentioning *dataset* in the name of the dataset
   * Avoid capital letters, special charaters (including spaces) in files and
     directories names. Spaces can be replaced by hyphens (``-``).
   * Multiple archives can be provided to spread the data (e.g. dataset splits,
     raw data, extra data, ...)


Digital Object Identifier (DOI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to get a DOI to reference the dataset. A DOI is a permanent
id/URL which prevents losing references of online scientific data.
https://figshare.com can be used to create a DOI:

* Go in `My Data`
* Create an item by clicking `Create new item`
* Check `Metadata record only` at the top
* Fill the metadata fields

Then reference the dataset using https://doi.org like this:
https://doi.org/10.6084/m9.figshare.2066037
