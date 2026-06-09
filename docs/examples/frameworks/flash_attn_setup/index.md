# Flash Attention Setup

The flash attention mechanism is a technique used in transformer models to reduce the computational complexity of the attention mechanism. It does this by using a more efficient algorithm for computing the attention scores, which allows it to handle longer sequences without running out of memory.

This package is a bit complex to install, as it requires an installation of PyTorch with CUDA support, as well as some additional dependencies. This example shows how to set up a development environment for flash attention using uv.

## Prerequisites

Make sure to read the following sections of the documentation before using this example:

* [Quick Start](../../../getting_started/)
* [Running your code](../../../userguides/running_code)
* [uv](../../../technical_reference/general_theory/portability/#uv)

Other resources:

* [Flash Attention GitHub repository](https://github.com/dao-ailab/flash-attention)

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
    You need to set the number of `MAX_JOBS` to 1 to avoid out of memory errors during the installation of flash attention because of too many parallel compilation jobs.

!!! warning
    `--no-build-isolation` is required to install flash attention because it needs to access the CUDA libraries that are available on the system.

!!! warning
    `FLASH_ATTENTION_SKIP_CUDA_BUILD=0` is required to ensure that flash attention is compiled with CUDA support.

!!! tip
    Adapt the value of `TORCH_CUDA_ARCH_LIST` to the compute capability of the GPU you are using. You can find the compute capability of your GPU on the [NVIDIA website](https://developer.nvidia.com/cuda-gpus). Setting this variable ensures that flash attention is compiled with support for your specific GPU architecture, which can improve performance and installation time.

    In this example, we set `TORCH_CUDA_ARCH_LIST` to "9.0" which corresponds to the compute capability of the NVIDIA H100 GPU. You can also set it to multiple values if you want to support multiple GPU architectures with : `TORCH_CUDA_ARCH_LIST="9.0;8.0"`.

**main.py**

```python
--8<-- "docs/examples/frameworks/flash_attn_setup/main.py"
```

## Running this example

This assumes that you already installed UV on the cluster you are working on.
TODO : with interactive job example

TODO : adapt module to the correct cuda version

```bash
# Get access to nvcc and the CUDA libraries
module load cuda/12.6

# Create the virtual environment and install all dependencies
uv sync
```