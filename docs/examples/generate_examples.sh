#!/bin/bash
set -e
patch  docs/examples/distributed/001_single_gpu/main.py \
    -i docs/examples/distributed/002_multi_gpu/main.py.patch \
    -o docs/examples/distributed/002_multi_gpu/main.py
patch  docs/examples/distributed/001_single_gpu/job.sh \
    -i docs/examples/distributed/002_multi_gpu/job.sh.patch \
    -o docs/examples/distributed/002_multi_gpu/job.sh

patch  docs/examples/distributed/002_multi_gpu/main.py \
    -i docs/examples/distributed/003_multi_node/main.py.patch \
    -o docs/examples/distributed/003_multi_node/main.py
patch  docs/examples/distributed/002_multi_gpu/job.sh \
    -i docs/examples/distributed/003_multi_node/job.sh.patch \
    -o docs/examples/distributed/003_multi_node/job.sh
