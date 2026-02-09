# Single GPU Job


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* [PyTorch setup](../../frameworks/pytorch_setup/index.md)

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu)

**job.sh**

```bash
--8<-- "docs/examples/distributed/single_gpu/job.sh"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/distributed/single_gpu/pyproject.toml"
```

**main.py**

```python
--8<-- "docs/examples/distributed/single_gpu/main.py"
```

## Running this example

```bash
 $ sbatch job.sh
```
