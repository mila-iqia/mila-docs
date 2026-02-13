# Jax


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* [JAX setup](../jax_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax)


**job.sh**

```bash
--8<-- "docs/examples/frameworks/jax/job.sh"
```

**main.py**

```python
--8<-- "docs/examples/frameworks/jax/main.py"
```

**model.py**

```python
--8<-- "docs/examples/frameworks/jax/model.py"
```

## Running this example

```bash
$ sbatch job.sh
```
