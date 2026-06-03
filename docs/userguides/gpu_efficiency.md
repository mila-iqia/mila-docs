---
title: "Could my job run faster? How to identify GPU waste"
description: >-
  Understand GPU efficiency metrics, diagnose underutilization in your jobs,
  and apply concrete best practices to improve throughput on the Mila cluster.
---

# Could my job run faster? How to identify GPU waste

## 1. Why should you care about GPU efficiency?

Optimizing your GPU usage directly accelerates research velocity. As compute
power at Mila is a shared resource, efficient jobs on the cluster bring a
two-sided advantage:

- **For you:** Eliminating bottlenecks speeds up your training times and helps
  unearth hidden bugs in your data loaders or model architectures. You get
  useful results faster.
- **For Mila:** Maximizing efficiency frees up cluster nodes, resulting in
  shorter queue times and more parallel experiments across the institute.

In other words, efficient compute utilization makes Mila research thrive.

## 2. The basics of compute efficiency: Utilization vs. Occupancy

A common pitfall is relying solely on `nvidia-smi`. The trap is that
`nvidia-smi` reports "100% Utilization" as soon as even one thread is active
on the GPU — like saying a bus is "used" because there is one passenger inside.

To better measure actual utilization, look at **Streaming Multiprocessor (SM)
Occupancy**: how many "seats" in the bus are actually filled. More context is
available in this
[external post on GPU utilization metrics](https://www.trainy.ai/blog/gpu-utilization-misleading).

Use the table below as a reference to evaluate your SM occupancy:

| SM Occupancy | Assessment |
|---|---|
| < 5% | Critical waste |
| ~10% | Poor utilization — the GPU is mostly waiting |
| ~30% | Good utilization |
| ≥ 50% | Great / optimized utilization |

## 3. How to diagnose your jobs today?

You can self-diagnose using these framework-agnostic methods.

### Method A: Weights & Biases (Recommended)

If you use WandB, go to the **System** tab of your run to find data on GPU
utilization, CPU usage, and memory. See
[Diagnose training bottlenecks](../userguides/wandb.md#diagnose-training-bottlenecks)
for details.

### Method B: The interactive check

During a job, you can `srun` into your allocated node and run:

```bash
# Basic check
nvidia-smi

# Better: Check SM occupancy in real time (requires dcgmi or nvitop)
# High power draw (Watts) is usually a good signal of active GPU utilization.
nvitop
```

## 4. Typical DOs and DON'Ts of compute efficiency

Even if situations are very diverse, the following guidelines pave the way for
efficient GPU utilization.

### ✅ DOs — To improve your runs

1. **Profile before scaling:** Run a test job with a profiler (WandB or
   TensorBoard) before launching large sweeps to ensure your data loader
   saturates the GPU.
2. **Optimize data pipelines:** Set `num_workers > 0` (2–4 per allocated GPU)
   and enable `pin_memory=True` in your PyTorch `DataLoader` to prevent GPU
   stalling.
3. **Implement checkpointing:** Save training states regularly so jobs can
   automatically resume after preemption or timeouts without losing previous
   compute hours.
4. **Right-size resource requests:** Use lower-tier nodes or MIG
   (Multi-Instance GPU) slices for small models or debugging instead of
   allocating full high-end nodes.

### ❌ DON'Ts — Common pitfalls

1. **Hoarding nodes:** Do not keep high-end GPUs (e.g., H100s) allocated on
   interactive partitions while away from your keyboard. Release them if not
   actively computing.
2. **Avoiding preemption queues:** Do not camp on non-preemptible partitions
   just to avoid writing checkpointing code — this tanks your overall queue
   priority.
3. **Over-allocating CPU cores:** Do not request excessive CPU cores (e.g., 40
   CPUs for 1 GPU) unless your preprocessing explicitly requires it. Mila
   provides CPU-only nodes if needed.
4. **Scaling GPUs to fix I/O bottlenecks:** Do not add more GPUs if your
   pipeline is bottlenecked by storage read latency or CPU preprocessing — you
   will just idle more hardware.
5. **Underutilizing VRAM:** Do not run with low batch sizes if your VRAM usage
   is under 20%. Increase the batch size to maximize SM occupancy.

## 5. Get help

If you have any questions, feel free to reach out on Slack
(`#mila-cluster`, `#compute-canada`), or drop by during
[Office Hours](https://docs.mila.quebec/help/office_hours/).
We are always happy to provide early guidance or share thoughts on how to
improve your experiments — in the interest of the community.

You can also use an LLM with
[curated Mila cluster context](https://docs.mila.quebec/ai/) to investigate
and remove obvious performance issues if you prefer to take matters into your
own hands.
