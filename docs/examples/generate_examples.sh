#!/bin/bash

function generate_example() {
    patch  docs/examples/$1 -i docs/examples/$2.diff -o docs/examples/$2
}

# single_gpu -> multi_gpu
generate_example distributed/001_single_gpu/job.sh distributed/002_multi_gpu/job.sh
generate_example distributed/001_single_gpu/main.py distributed/002_multi_gpu/main.py

# multi_gpu -> multi_node
generate_example distributed/002_multi_gpu/job.sh distributed/003_multi_node/job.sh
generate_example distributed/002_multi_gpu/main.py distributed/003_multi_node/main.py
