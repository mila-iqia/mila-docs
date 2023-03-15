
# Launching the example:


```bash
sbatch --reservation=milabench --nodes=2 --ntasks-per-node=2 job.sh
sbatch --reservation=milabench --nodes=1 --ntasks-per-node=4 job.sh
sbatch --reservation=milabench --nodes=2 --ntasks-per-node=2 \
    --export=ALL,ACCELERATE_CONFIG=ds_level3.yaml job.sh
sbatch --reservation=milabench --nodes=1 --ntasks-per-node=4 \
    --export=ALL,ACCELERATE_CONFIG=ds_level3.yaml job.sh
```


## TOTAL VRAM required to train each model

| Model    | VRAM       |
| -------- | ---------- |
| OPT-125m | 3.14 GB    |
| OPT-350m | 8.32 GB    |
| OPT-1.3B | 33.07 GB   |
| OPT-2.7B | 66.66 GB   |
| OPT-6.7B | 167.42 GB  |
| OPT-13B  | 320.96 GB  |
| OPT-30B  | 740.67 GB  |
| OPT-66B  | 1629.47 GB |
| OPT-175B | 4320.56 GB |

# NOTE: about `generate` with DeepSpeed level=3
- Need to pass syneced_gpus=True to `.generate()` method, otherwise there's a stall
  

- Look into: "overlap_comm": true,