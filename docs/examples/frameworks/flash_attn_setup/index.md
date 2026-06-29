---
title: Flash Attention Setup
description: Install Flash Attention on the Mila cluster using uv, from a
  pre-built wheel or compiled from source.
---

# Flash Attention Setup

Flash Attention is a memory-efficient attention algorithm for transformer
models that reduces memory usage for long sequences. Installing it
requires PyTorch with CUDA support and careful dependency management.
This guide covers two installation paths using uv: from a pre-built
wheel (faster, no compilation), or built from source when no compatible
wheel is available.

## Prerequisites

Make sure to read the following sections of the documentation before using this example:

* [Quick Start](../../../getting_started/train_first_model/)
* [Running your code](../../../userguides/running_code)
* [uv](../../../technical_reference/general_theory/portability/#uv)

Other resources:

* [Flash Attention GitHub repository](https://github.com/dao-ailab/flash-attention)

## Installation

Flash Attention provides pre-built wheels for common combinations of
CUDA, PyTorch, and Python on the
[Flash Attention releases page](https://github.com/Dao-AILab/flash-attention/releases).
If a compatible wheel is listed, use the [From a pre-built wheel](#from-a-pre-built-wheel) section.
Otherwise, use [Building from source](#building-from-source) to compile Flash Attention on a compute node.

### From a pre-built wheel

Pre-built wheels are available for common CUDA, PyTorch, and Python
combinations. Wheel filenames encode the target configuration, like :

```
flash_attn-2.8.3.post1+cu126torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

This wheel targets:

* `flash_attn-2.8.3.post1` — flash attention `2.8.3.post1`
* `cu126` — CUDA 12.6
* `torch2.7` — PyTorch 2.7
* `cp312` — Python 3.12
* `linux_x86_64` — Linux x86_64

!!! tip
    Pre-built wheels require no compilation and can be installed on login nodes.

Add the wheel URL as a dependency source in `pyproject.toml`,
replacing the URL with the one matching the target configuration:

```toml title="from_pre_build_wheel/pyproject.toml"
--8<-- "docs/examples/frameworks/flash_attn_setup/from_pre_build_wheel/pyproject.toml"
```

Install the dependencies:

```bash
# Create the virtual environment and install all dependencies
uv sync
```

??? tip "Reusing a locally built wheel"
    A wheel built from source (see the[Building from source](#building-from-source))
    can also be used. In the sources section of the `pyproject.toml`,
    replace `url` by `path`, like:

    ```toml
    [tool.uv.sources]
    flash-attn = { path = "<path_to_wheel>" }
    ```

### Building from source
Build Flash Attention from source when no pre-built wheel matches
the target CUDA, PyTorch, and Python combination.

!!! warning
    Building from source requires significant memory and time. Do
    not run the build on login nodes, submit a dedicated job to a
    compute node instead.

Use the `pyproject.toml` configuration:

```toml title="build_from_source/pyproject.toml"
--8<-- "docs/examples/frameworks/flash_attn_setup/build_from_source/pyproject.toml"
```

!!! warning
    `MAX_JOBS = "4"` limits parallel compilation to prevent
    out-of-memory errors during the build. Increase this value only
    when the compute node has sufficient memory for more parallel jobs.

    `FLASH_ATTENTION_SKIP_CUDA_BUILD = "0"` ensures that Flash Attention
    is compiled with CUDA support.

!!! tip
    Adapt `TORCH_CUDA_ARCH_LIST` to the compute capability of the
    target GPU. Find compute capabilities on the
    [NVIDIA website](https://developer.nvidia.com/cuda-gpus). `"9.0"`
    targets the H100. To support multiple architectures, separate
    values with semicolons: `TORCH_CUDA_ARCH_LIST="9.0;8.0;..."`.

Submit the build using the job script:

```bash title="build_from_source/job.sh"
--8<-- "docs/examples/frameworks/flash_attn_setup/build_from_source/job.sh"
```

```bash
sbatch build_from_source/job.sh
```

Once the build completes, uv caches the compiled wheel, which can be reused in other projects without recompiling.
To reuse the wheel directly, (for example, on an other cluster), locate the cached wheel with:

```bash
find ~/.cache/uv -name "flash_attn*.whl" 2>/dev/null
```

## Example

The full source code for this example is available on
[the mila-docs GitHub repository](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/flash_attn_setup).

=== "Job script"
    ```bash title="job.sh"
    --8<-- "docs/examples/frameworks/flash_attn_setup/job.sh"
    ```
=== "Python code"
    ```python title="main.py"
    --8<-- "docs/examples/frameworks/flash_attn_setup/main.py"
    ```

## Running this example

Submit the job with sbatch:

```bash
sbatch job.sh
```
