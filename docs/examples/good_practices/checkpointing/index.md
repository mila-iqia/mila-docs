### Checkpointing


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* [PyTorch Setup](../../frameworks/pytorch_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)

The full source code for this example is available on [the mila-docs GitHub
repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/checkpointing)


**job.sh**

```diff
--8<-- "docs/examples/good_practices/checkpointing/job.sh.diff"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/good_practices/checkpointing/pyproject.toml"
```

**main.py**

```diff
--8<-- "docs/examples/good_practices/checkpointing/main.py.diff"
```

#### Running this example

```bash
$ sbatch job.sh
```
