# Flash Attention Setup

The flash attention mechanism is a technique used in transformer models to 
reduce the computational complexity of the attention mechanism. It does this by
using a more efficient algorithm for computing the attention scores, which allows
it to handle longer sequences without running out of memory.

This package is a bit complex to install, as it requires an installation of
PyTorch with CUDA support, as well as some additional dependencies. This example 
shows how to set up a development environment for flash attention using uv.

## Prerequisites

Make sure to read the following sections of the documentation before using this example:

* [Quick Start](../../../getting_started/)
* [Running your code](../../../userguides/running_code)
* [uv](../../../technical_reference/general_theory/portability/#uv)
* [Modules](../../../technical_reference/general_theory/portability/#using-modules)

Other resources:

* [Flash Attention GitHub repository](https://github.com/dao-ailab/flash-attention)

## Installation
By default, flash attention will try to find a pre-compiled version of the library that matches your system configuration (PyTorch and CUDA). If it cannot find one, it will attempt to compile the library from source.

This assumes that you already installed UV on the cluster you are working on.
Before installing the dependencies, make sure to load the appropriate CUDA module.

=== "From a pre-buil wheel"
    TODO, how to install from a pre-build wheel.
    (check releases from official github repo, https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1). If a wheel is not available, consider the step "Building from source"

    TODO : Can be install on the login nodes.

    TODO : Add pyproject config.

    For example, on the Mila cluster, you can load the `cuda/12.6` module:

    ```bash
    # Get access to the CUDA libraries
    module load cuda/12.6

    # Create the virtual environment and install all dependencies
    uv sync
    ```

    TODO : other possible sources.

    !!! warning
        You need to set the number of `MAX_JOBS` to 1 to avoid out of memory 
        errors during the installation of flash attention because of too many 
        parallel compilation jobs.


=== "Building from source"
    TODO : not all version of CUDA / Python have a prebuild wheel available.
    TODO : how to build the package from sources.
    TODO : Should not be installed from the login nodes (very long and need a lot of memory). Need to reserved compute nodes.

    TODO : Add pyproject config.

    **job_build_from_source.sh**

    ```bash
    --8<-- "docs/examples/frameworks/flash_attn_setup/job_build_from_source.sh"
    ```

    TODO : tip : value of MAX_JOBS can be adapted

    !!! tip
        Adapt the value of `TORCH_CUDA_ARCH_LIST` to the compute capability of the 
        GPU you are using. You can find the compute capability of your GPU on the 
        [NVIDIA website](https://developer.nvidia.com/cuda-gpus). Setting this variable 
        ensures that flash attention is compiled with support for your specific GPU architecture, 
        which can improve performance and installation time.

        In this example, we set `TORCH_CUDA_ARCH_LIST` to "8.9" which corresponds to 
        the compute capability of the NVIDIA L40S GPU. You can also set it to multiple 
        values if you want to support multiple GPU architectures with : 
        `TORCH_CUDA_ARCH_LIST="9.0;8.0;..."`.

    Then, you can submit a job to build the package :
    ```bash
    $ sbatch job.sh
    ```

    TODO : When the wheel is builded, you can re-use it :
    TODO : Add command to find the builded wheel.
    ```bash
    $ find ~/.cache/uv -name "flash_attn*.whl" 2>/dev/null
    ```
    TODO : Add uv command to install from a wheel.
    ```bash
    $ uv add <path_to_wheel>
    ```
    TODO : or add in your `pyproject.toml` with https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-sources :
    ```toml
    [tool.uv.sources]
    flash-attn = { path = "<path_to_wheel>" }
    ```

## Example

The full source code for this example is available on [the mila-docs GitHub repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/flash_attn)

**job.sh**

```bash
--8<-- "docs/examples/frameworks/flash_attn_setup/job.sh"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/frameworks/flash_attn_setup/pyproject.toml"
```

!!! warning
    `--no-build-isolation` is required to install flash attention because 
    it needs to access the CUDA libraries that are available on the system.

!!! warning
    `FLASH_ATTENTION_SKIP_CUDA_BUILD=0` is required to ensure that flash 
    attention is compiled with CUDA support. 

**main.py**

```python
--8<-- "docs/examples/frameworks/flash_attn_setup/main.py"
```

## Running this example

You can submit a job to run the example with sbatch:

```bash
 $ sbatch job.sh
```