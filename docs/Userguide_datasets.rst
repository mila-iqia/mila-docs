Contributing datasets
=====================


If a dataset could help the research of others at Mila, `this form
<https://forms.gle/vDVwD2rZBmYHENgZA>`_ can be filled to request its addition
to `/network/datasets <Information.html#storage>`_.

Those datasets can be mirrored to the Béluga cluster in
``~/projects/rrg-bengioy-ad/data/curated/`` if they follow Compute Canada's
good practices on data.


Publicly share a Mila dataset
-----------------------------


Mila offers two ways to publicly share a Mila dataset:

* `Academic Torrent <https://academictorrents.com>`_
* `Google Drive
  <https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_

Note that these options are not mutually exclusive and both can be used.

We do host Mila datasets using Academic Torrent and we also offer Google Drive
space.

.. note::
   * Avoid mentioning *dataset* in the name of the dataset
   * Avoid capital letters, special charaters (including spaces) in files and
     directories names. Spaces can be replaced by hyphens (``-``).
   * Multiple archives can be provided to spread the data (e.g. dataset splits,
     raw data, extra data, ...)


Academic Torrent
^^^^^^^^^^^^^^^^


Mila hosts/seeds some datasets created by the Mila community through `Academic
Torrent <https://academictorrents.com>`_. The first step is to create `an
account and a torrent file <https://academictorrents.com/upload.php>`_.

Then drop the dataset in ``/miniscratch/transit_datasets`` and send the
Academic Torrent url to `Mila's helpdesk <https://it-support.mila.quebec>`_. If
the dataset does not reside on the Mila clusterm, only the Academic Torrent url
would be needed to proceed with the initial download. Then you can delete /
stop sharing your copy.


Google Drive
^^^^^^^^^^^^


Only a member of the staff team can upload to `Mila's Google Drive
<https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_
which requires to first drop the dataset in ``/miniscratch/transit_datasets``.
Then, contact `Mila's helpdesk <https://it-support.mila.quebec>`_ and provide
the following informations:

* directory containing the archived dataset (zip is favored) in
  ``/miniscratch/transit_datasets``
* the name of the dataset
* md5 checksum of the archive
* the arXiv and GitHub urls (those can be sent later if the article is still in
  the submission process)
* instructions to know if the dataset needs to be ``unzip``\ed, ``untar``\ed or
  else before uploading to Google Drive


Digital Object Identifier (DOI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


It is recommended to get a DOI to reference the dataset. A DOI is a permanent
id/url which prevents losing references of online scientific data.
https://figshare.com can be used to create a DOI:

* Go in `My Data`
* Create an item by clicking `Create new item`
* Check `Metadata record only` at the top
* Fill the metadata fields

Then reference the dataset using https://doi.org like this:
https://doi.org/10.6084/m9.figshare.2066037
