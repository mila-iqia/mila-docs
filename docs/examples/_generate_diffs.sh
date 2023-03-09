#!/bin/bash
# Use this to update the diffs based on the contents of the files.
# NOTE: I (@lebrice) wouldn't recommend editing the files directly, since it might break things
# that depend on it.

set -e

git diff --no-index \
    docs/examples/distributed/001_single_gpu/job.sh \
    docs/examples/distributed/002_multi_gpu/job.sh \
    > docs/examples/distributed/002_multi_gpu/job.sh.patch
git diff --no-index \
    docs/examples/distributed/001_single_gpu/main.py \
    docs/examples/distributed/002_multi_gpu/main.py \
    > docs/examples/distributed/002_multi_gpu/main.py.patch

git diff --no-index \
    docs/examples/distributed/002_multi_gpu/job.sh \
    docs/examples/distributed/003_multi_node/job.sh \
    > docs/examples/distributed/003_multi_node/job.sh.patch
git diff --no-index \
    docs/examples/distributed/002_multi_gpu/main.py \
    docs/examples/distributed/003_multi_node/main.py \
    > docs/examples/distributed/003_multi_node/main.py.patch
