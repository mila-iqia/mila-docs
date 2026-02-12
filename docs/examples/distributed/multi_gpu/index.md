# Multi-GPU Job


Prerequisites:

* [PyTorch Setup](../../frameworks/pytorch_setup/index.md)
* [Single-GPU Job](../single_gpu/index.md)

Other interesting resources:

* [sebarnold.net dist blog](https://sebarnold.net/dist_blog/)
* [Multi-node PyTorch distributed training guide (Lambda Labs)](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)

Click here to see [the code for this example](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/multi_gpu)

**job.sh**

```diff
--8<-- "docs/examples/distributed/multi_gpu/job.sh.diff"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/distributed/multi_gpu/pyproject.toml"
```

**main.py**

```diff
--8<-- "docs/examples/distributed/multi_gpu/main.py.diff"
```

## Running this example

```bash
sbatch job.sh
```
