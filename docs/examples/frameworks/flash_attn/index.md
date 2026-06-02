# Flash Attention

TODO : context

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
--8<-- "docs/examples/frameworks/flash_attn/job.sh"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/frameworks/flash_attn/pyproject.toml"
```

!!! warning
    You need to set the number of MAX_JOBS to 1 in the `pyproject.toml` file to avoid out of memory errors during the installation of flash attention. This is because flash attention requires a lot of memory to compile, and setting MAX_JOBS to 1 ensures that only one job is running at a time during the installation process.

**main.py**

```python
--8<-- "docs/examples/frameworks/flash_attn/main.py"
```

## Running this example

This assumes that you already installed UV on the cluster you are working on.
TODO