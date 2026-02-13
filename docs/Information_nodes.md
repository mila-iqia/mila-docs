# Node profile description

<!-- Je trouve cela un peu futile de maintenir cette documentation à jour
manuellement.  Peut-être pourrions nous créer dans ce dossier des sripts qui
pourraient créer une entrée RST et qui pourraient être exécutés sur un noeud au
Mila pour les mises à jour. -->
<!-- TODO: Maybe add the tablesort feature of mkdocs: https://squidfunk.github.io/mkdocs-material/reference/data-tables/#sortable-tables -->

| Name                  | GPU Model | Mem | #   | CPUs | Sockets | Cores/Socket | Threads/Core | Memory (GB) | TmpDisk (TB) | Arch   | Slurm Features         |
| --------------------- | --------- | --- | --- | ---- | ------- | ------------ | ------------ | ----------- | ------------ | ------ | ---------------------- |
| **GPU Compute Nodes** |           |     |     |      |         |              |              |             |              |        |                        |
| **cn-a[001-011]**     | RTX8000   | 48  | 8   | 40   | 2       | 20           | 1            | 384         | 3.6          | x86_64 | turing,48gb            |
| **cn-b[001-005]**     | V100      | 32  | 8   | 40   | 2       | 20           | 1            | 384         | 3.6          | x86_64 | volta,nvlink,32gb      |
| **cn-c[001-040]**     | RTX8000   | 48  | 8   | 64   | 2       | 32           | 1            | 384         | 3            | x86_64 | turing,48gb            |
| **cn-g[001-029]**     | A100      | 80  | 4   | 64   | 2       | 32           | 1            | 1024        | 7            | x86_64 | ampere,nvlink,80gb     |
| **cn-i001**           | A100      | 80  | 4   | 64   | 2       | 32           | 1            | 1024        | 3.6          | x86_64 | ampere,80gb            |
| **cn-j001**           | A6000     | 48  | 8   | 64   | 2       | 32           | 1            | 1024        | 3.6          | x86_64 | ampere,48gb            |
| **cn-k[001-004]**     | A100      | 40  | 4   | 48   | 2       | 24           | 1            | 512         | 3.6          | x86_64 | ampere,nvlink,40gb     |
| **cn-l[001-091]**     | L40S      | 48  | 4   | 48   | 2       | 24           | 1            | 1024        | 7            | x86_64 | lovelace,48gb          |
| **cn-n[001-002]**     | H100      | 80  | 8   | 192  | 2       | 96           | 1            | 2048        | 35           | x86_64 | hopper,nvlink,80gb     |
| **DGX Systems**       |           |     |     |      |         |              |              |             |              |        |                        |
| **cn-d[001-002]**     | A100      | 40  | 8   | 128  | 2       | 64           | 1            | 1024        | 14           | x86_64 | ampere,nvlink,dgx,40gb |
| **cn-d[003-004]**     | A100      | 80  | 8   | 128  | 2       | 64           | 1            | 2048        | 28           | x86_64 | ampere,nvlink,dgx,80gb |
| **cn-e[002-003]**     | V100      | 32  | 8   | 40   | 2       | 20           | 1            | 512         | 7            | x86_64 | volta,nvlink,dgx,32gb  |
| **CPU Compute Nodes** |           |     |     |      |         |              |              |             |              |        |                        |
| **cn-f[001-004]**     | -         | -   | -   | 32   | 1       | 32           | 1            | 256         | 10           | x86_64 | rome                   |
| **cn-h[001-004]**     | -         | -   | -   | 64   | 2       | 32           | 1            | 768         | 7            | x86_64 | milan                  |
| **cn-m[001-004]**     | -         | -   | -   | 96   | 2       | 48           | 1            | 1024        | 7            | x86_64 | sapphire               |

## Special nodes and outliers


### DGX A100


DGX A100 nodes are NVIDIA appliances with 8 NVIDIA A100 Tensor Core GPUs. Each
GPU has either 40 GB or 80 GB of memory, for a total of 320 GB or 640 GB per
appliance. The GPUs are interconnected via 6 NVSwitches which allow for 600 GB/s
point-to-point bandwidth (unidirectional) and a full bisection bandwidth of 4.8
TB/s (bidirectional). See the table above for the specifications of each
appliance.

In order to run jobs on a DGX A100 with 40GB GPUs, add the flags below to your
Slurm commands:

```bash
--gres=gpu:a100:<number> --constraint="dgx&ampere"
```

In order to run jobs on a DGX A100 with 80GB GPUs, add the flags below to your
Slurm commands:

```bash
--gres=gpu:a100l:<number> --constraint="dgx&ampere"
```
