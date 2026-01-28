### Contributing datasets

If a dataset could help the research of others at Mila, `this form
<https://forms.gle/vDVwD2rZBmYHENgZA>`_ can be filled to request its addition
to [/network/datasets ](Information.html#storage).

#### Publicly share a Mila dataset

Mila offers two ways to publicly share a Mila dataset:

* [Academic Torrent ](https://academictorrents.com)
* `Google Drive
  <https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_

Note that these options are not mutually exclusive and both can be used.

##### Academic Torrent

Mila hosts/seeds some datasets created by the Mila community through `Academic
Torrent <https://academictorrents.com>`_. The first step is to create `an
account and a torrent file <https://academictorrents.com/upload.php>`_.

Then drop the dataset in `/network/scratch/.transit_datasets` and send the
Academic Torrent URL to [Mila's helpdesk ](https://it-support.mila.quebec). If
the dataset does not reside on the Mila cluster, only the Academic Torrent URL
would be needed to proceed with the initial download. Then you can delete /
stop sharing your copy.

> **NOTE**
> * Avoid mentioning *dataset* in the name of the dataset
> * Avoid capital letters, special charaters (including spaces) in files and
>   directories names. Spaces can be replaced by hyphens (``-``).
> * Multiple archives can be provided to spread the data (e.g. dataset splits,
>   raw data, extra data, ...)

###### Generate a .torrent file to be uploaded to Academic Torrent

The command line / Python utility `torrentool
<https://github.com/idlesign/torrentool>`_ can be used to create a
`DATASET_NAME.torrent` file:

.. code-block:: sh

   # Install torrentool
   python3 -m pip install torrentool click
   # Change Directory to the location of the dataset to be hosted by Mila
   cd /network/scratch/.transit_datasets
   torrent create --tracker https://academictorrents.com/announce.php DATASET_NAME

The resulting `DATASET_NAME.torrent` can then be used to register a new dataset
on Academic Torrent.

> **WARNING**
> * The creation of a `DATASET_NAME.torrent` file requires the computation of
>   checksums for the dataset content which can quickly become CPU-heavy. This
>   process should *not* be executed on a login node

###### Download a dataset from Academic Torrent

Academic Torrent provides a `Python API
<https://github.com/academictorrents/at-python>`_ to easily download a dataset
from it's registered list:

.. code-block:: python

   # Install the Python API with:
   # python3 -m pip install academictorrents
   import academictorrents as at
   mnist_path = at.get("323a0048d87ca79b68f12a6350a57776b6a3b7fb", datastore="~/scratch/.academictorrents-datastore") # Download the mnist dataset

> **NOTE**
> Current needs have been evaluated to be for a download speed of about 10
> MB/s. This speed can be higher if more users also seeds the dataset.

##### Google Drive

Only a member of the staff team can upload to `Mila's Google Drive
<https://drive.google.com/drive/folders/1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt>`_
which requires to first drop the dataset in
`/network/scratch/.transit_datasets`. Then, contact `Mila's helpdesk
<https://it-support.mila.quebec>`_ and provide the following informations:

* directory containing the archived dataset (zip is favored) in
  `/network/scratch/.transit_datasets`
* the name of the dataset
* a licence in `.txt` format. One of the `the creative common
  <https://creativecommons.org/about/cclicenses/>`_ licenses can be used. It is
  recommended to at least have the *Attribution* option. The *No Derivatives*
  option is discouraged unless the dataset should not be modified by others.
* MD5 checksum of the archive
* the arXiv and GitHub URLs (those can be sent later if the article is still in
  the submission process)
* instructions to know if the dataset needs to be `unzip`\ed, `untar`\ed or
  else before uploading to Google Drive

> **NOTE**
> * Avoid mentioning *dataset* in the name of the dataset
> * Avoid capital letters, special charaters (including spaces) in files and
>   directories names. Spaces can be replaced by hyphens (``-``).
> * Multiple archives can be provided to spread the data (e.g. dataset splits,
>   raw data, extra data, ...)

###### Download a dataset from Mila's Google Drive with  ``gdown``

The utility [gdown ](https://github.com/wkentaro/gdown) is a simple utility to
download data from Google Drive from the command line shell or in a Python
script and requires no setup.

> **WARNING**
> A limitation however is that it uses a shared client id which can cause a
> quota block when too many users uses it in the same day. It is described in
> a `GitHub issue
> <https://github.com/wkentaro/gdown/issues/43#issuecomment-642182100>`_.

###### Download a dataset from Mila's Google Drive with ``rclone``

[Rclone ](https://rclone.org/) is a command line program to manage files on
cloud storage. In the context of a Google Drive remote, it allows to specify a
client id to avoid sharing with other users which avoid quota limits. Rclone
describes the creation of a `client id in its documentaton
<https://rclone.org/drive/#making-your-own-client-id>`_. Once this is done, a
remote for Mila's Google Drive can be configured from the command line:

.. code-block:: sh

   rclone config create mila-gdrive drive client_id XXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.apps.googleusercontent.com \
       client_secret XXXXXXXXXXXXX-XXXXXXXXXX \
       scope 'drive.readonly' \
       root_folder_id 1peJ6VF9wQ-LeETgcdGxu1e4fo28JbtUt \
       config_is_local false \
       config_refresh_token false

The remote can then be used to download a dataset:

.. code-block:: sh

   rclone copy --progress mila-gdrive:DATASET_NAME/ ~/scratch/datasets/DATASET_NAME/

Rclone is available from the `conda channel conda-forge
<https://anaconda.org/conda-forge/rclone>`_.

##### Digital Object Identifier (DOI)

It is recommended to get a DOI to reference the dataset. A DOI is a permanent
id/URL which prevents losing references of online scientific data.
https://figshare.com can be used to create a DOI:

* Go in `My Data`
* Create an item by clicking `Create new item`
* Check `Metadata record only` at the top
* Fill the metadata fields

Then reference the dataset using https://doi.org like this:
https://doi.org/10.6084/m9.figshare.2066037