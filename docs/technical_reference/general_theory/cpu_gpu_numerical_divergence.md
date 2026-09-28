---
title: "Why CPU and GPU Training Runs Diverge Numerically"
description: >-
  Understand why running the same model on CPU or GPU can lead to different results.
---

# Why CPU and GPU Training Runs Diverge Numerically

## Summary

Running the *same* model, data, seed, and code on CPU vs. GPU (e.g. via the Pytorch function `tensor.to(device)`) will **not** produce bit-identical results. Losses typically match closely at the start of training and then drift apart, eventually diverging completely. This is expected behavior, not a bug — it stems from the properties of floating-point arithmetic that make it an imperfect simulation of arithmetic on real numbers.

## Root Cause

- IEEE `float32` is **the same format** on both CPU and GPU — there is no inherent precision difference between devices. Likewise, `float64` numbers are interpreted by both CPU and GPU identically.
- Floating-point addition is **non-associative**: `(a + b) + c` is not guaranteed to equal `a + (b + c)` due to rounding at each step.
- CPUs and GPUs parallelize reductions (sums, matmuls, etc.) differently to fit their respective hardware, so they add the same set of numbers **in a different order**.
- Different summation order → different rounding error at each step → tiny discrepancies that compound over many operations and training steps, eventually causing full trajectory divergence (a chaotic, butterfly-effect-style process).

!!! tip "Takeaway"
    Never rely on tensor math being bit-exact across devices, or even across different thread counts on the same device.

## Mitigation Strategies (and their costs)

| Strategy | Effect | Cost / Caveat |
|---|---|---|
| Use `float64` instead of `float32` | Reduces rounding error per operation, often "fixes" the visible divergence in practice | On **CPU**: reasonable — x86 does float64 addition at roughly half the throughput of float32. On **GPU**: usually a bad idea — most consumer/data-center GPUs have float64 throughput at only 1/24, 1/32, or even 1/64 of float32 (only workstation-grade cards like some Teslas get closer to 1/2). |
| Use `torch.sum()`'s precision argument | Lets you request higher internal accumulation precision without changing the whole model's dtype | Still subject to the general CPU/GPU order-of-operations caveat above, but much less sensitive to it. |
| Accept the divergence | Often the right call for training runs where exact reproducibility isn't required | Only track/compare aggregate metrics (final loss, accuracy) across devices rather than exact per-step trajectories. |
