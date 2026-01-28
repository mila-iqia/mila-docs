### Data Transmission

Multiple methods can be used to transfer data to/from the cluster:

* `rsync --bwlimit=10mb`; this is the favored method since the bandwidth can
  be limited to prevent impacting the usage of the cluster: `rsync
  <https://cl-cheat-sheet.readthedocs.io/en/latest/#rsync>`_
* Digital Research Alliance of Canada: [Globus ](https://docs.alliancecan.ca/wiki/Globus)