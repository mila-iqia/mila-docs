# Launching the example:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-task=a100:4 --cpus-per-task=64 \
  --reservation=milabench-"$(date +%d)" --mem=1400G --export=ALL,ACCELERATE_CONFIG=configs/ds_level2.yaml \
  job.sh \
  --wandb_tags=mila --config_name=facebook/opt-13b --per_device_train_batch_size=1
```

## TODOS

- Add instructions on how to run the example outside a SLURM cluster
  - Need to set the `MASTER_ADDR` and `MASTER_PORT`, `WORLD_SIZE`, etc. environment variables manually, and call `accelerate launch` once per node with the right `--machine_index`.
- Look into: "overlap_comm": true
- Run the job using different configurations:
  - Changing models
  - Changing the accelerate config: ds_level2 (current), ds_level3, fsdp
  - Changing the batch size per device.

## TOTAL VRAM required to train each model

| Model    | VRAM  (params + optimizer + activations) (fp16 precision) |
| -------- | --------------------------------------------------------- |
| OPT-125m | 3.14 GB                                                   |
| OPT-350m | 8.32 GB                                                   |
| OPT-1.3B | 33.07 GB                                                  |
| OPT-2.7B | 66.66 GB                                                  |
| OPT-6.7B | 167.42 GB                                                 |
| OPT-13B  | 320.96 GB                                                 |
| OPT-30B  | 740.67 GB                                                 |
| OPT-66B  | 1629.47 GB                                                |
| OPT-175B | 4320.56 GB                                                |
