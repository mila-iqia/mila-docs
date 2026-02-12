# Wandb Setup


Prerequisites:

* [PyTorch setup](../../frameworks/pytorch_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)

Make sure to create a Wandb account, then you can either:

* Set your `WANDB_API_KEY` environment variable
* Run `wandb login` from the command line

Other resources:

* [Wandb quickstart](https://docs.wandb.ai/quickstart)

Click here to see [the source code for this example](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/wandb_setup).

**job.sh**

```diff
--8<-- "docs/examples/good_practices/wandb_setup/job.sh.diff"
```

**main.py**

```diff
--8<-- "docs/examples/good_practices/wandb_setup/main.py.diff"
```

## Running this example

**Note:** On DRAC clusters you will need to run `wandb off` to log your data as offline mode.
You will then be able to upload your runs with the command `wandb sync --sync-all`.

```bash
$ wandb login
```

```bash
$ sbatch job.sh
```
