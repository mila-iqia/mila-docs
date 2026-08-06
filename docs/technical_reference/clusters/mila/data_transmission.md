# Data Transmission

Multiple methods can be used to transfer data to/from the cluster.

## Rsync
The [rsync](https://linux.die.net/man/1/rsync) command is a fast and versatile
file copying tool that can be used to transfer and synchronize files between a
local machine and a remote cluster. It is particularly useful for transferring
large datasets or directories, as it only transfers the differences between the
source and destination files.

This is the favored method since the bandwidth can be limited to prevent impacting
the usage of the cluster:
```bash
rsync --bwlimit=10mb
```

## Globus Connect Personal
Mila doesn't own a Globus license but if the source or destination provides a
Globus account, [like Digital Research Alliance of Canada](https://docs.alliancecan.ca/wiki/Globus)
for example, it's possible to setup Globus Connect Personal to create a personal
endpoint on the Mila cluster by following the Globus guide to
[Install, Configure, and Uninstall Globus Connect Personal for Linux](https://docs.globus.org/how-to/globus-connect-personal-linux/).

This endpoint can then be used to transfer data to and from the Mila cluster.