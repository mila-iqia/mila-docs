# Single GPU Job with Jax

## Prerequisites

Make sure to read the following sections of the documentation before using this
example:

* [Jax setup](../jax_setup/index.md)
* [Single-GPU Job](../../distributed/single_gpu/index.md)

## Example

The full source code for this example is available on [the mila-docs GitHub repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu/single_gpu_jax)

**job.sh**

```bash
--8<-- "docs/examples/distributed/single_gpu/single_gpu_jax/job.sh"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/distributed/single_gpu/single_gpu_jax/pyproject.toml"
```

**main.py**

```python
--8<-- "docs/examples/distributed/single_gpu/single_gpu_jax/main.py"
```

**model.py**

```python
--8<-- "docs/examples/distributed/single_gpu/single_gpu_jax/model.py"
```

## Running this example

```bash
$ sbatch job.sh
```
