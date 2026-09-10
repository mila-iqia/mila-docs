---
title: "Why CPU and GPU Training Runs Diverge Numerically"
description: >-
  Understand how running the same model on CPU or GPU can lead to different results.
---

# Why CPU and GPU Training Runs Diverge Numerically

## Summary

Running the *same* model, data, seed, and code on CPU vs. GPU (e.g. via the Pytorch function `tensor.to(device)`) will **not** produce bit-identical results. Losses typically match closely at the start of training and then drift apart, eventually diverging completely. This is expected behavior, not a bug — it stems from floating-point arithmetic itself, not from a precision difference between devices.

## Root Cause

- IEEE float32/float64 are **the same format** on CPU and GPU — there is no inherent precision difference between devices.
- Floating-point addition is **non-associative**: `(a + b) + c` is not guaranteed to equal `a + (b + c)` due to rounding at each step.
- CPUs and GPUs parallelize reductions (sums, matmuls, etc.) differently to fit their respective hardware, so they add the same set of numbers **in a different order**.
- Different summation order → different rounding error at each step → tiny discrepancies that compound over many operations and training steps, eventually causing full trajectory divergence (a chaotic, butterfly-effect-style process).

!!! tip "Takeaway"
    Never rely on tensor math being bit-exact across devices, or even across different thread counts on the same device.

## Why GPUs Are Often (Slightly) More Numerically Stable

- Sequential (naive, one-at-a-time) summation accumulates rounding error linearly with the number of terms.
- **Pairwise summation** (summing in a tree/divide-and-conquer pattern) accumulates error more slowly — GPUs tend to use pairwise-style reductions to exploit parallelism, which incidentally makes them somewhat better-conditioned at a given precision than a naive sequential CPU sum.
- This advantage matters most for **ill-conditioned sums**, i.e. when:
    - the sum of absolute values is much larger than the absolute value of the sum (heavy cancellation), and/or
    - the ratio between the largest and smallest terms being summed is very large.
- For well-conditioned sums, the summation order barely matters and CPU vs. GPU differences stay negligible for longer.
- This is a heuristic tendency, not a guarantee — don't design correctness-critical code around it.

## Mitigation Strategies (and their costs)

| Strategy | Effect | Cost / Caveat |
|---|---|---|
| Use `float64` instead of `float32` | Reduces rounding error per operation, often "fixes" the visible divergence in practice | On **CPU**: reasonable — x86 does float64 addition at roughly half the throughput of float32. On **GPU**: usually a bad idea — most consumer/data-center GPUs have float64 throughput at only 1/24, 1/32, or even 1/64 of float32 (only workstation-grade cards like some Teslas get closer to 1/2). |
| Sort terms before summing (smallest → largest), or use a priority queue for streaming sums | Improves conditioning of sequential summation, closing some of the gap with pairwise summation | Extra sort/heap overhead; only helps when the sum is ill-conditioned to begin with. |
| Use `torch.sum()`'s precision argument | Lets you request higher internal accumulation precision without changing the whole model's dtype | Still subject to the general CPU/GPU order-of-operations caveat above. |
| Accept the divergence | Often the right call for training runs where exact reproducibility isn't required | Only track/compare aggregate metrics (final loss, accuracy) across devices rather than exact per-step trajectories. |

## Practical Guidance

1. If you see losses that match early in training and drift apart later when switching CPU ↔ GPU, this is very likely just accumulated floating-point rounding — check datatype, seeds, and code parity as a sanity check, but don't expect them to fix it.
2. Don't chase bit-exact reproducibility across devices or thread counts as a goal in itself; instead, define what "close enough" means for your use case (e.g. metrics within some tolerance).
3. If you need better numerical stability and can afford the cost, try `float64` **on CPU** first — it's the cheapest reliable fix. Avoid `float64` on GPU unless you're on stability-oriented hardware, since the throughput penalty is severe.
4. If you suspect ill-conditioned sums specifically (large dynamic range or heavy cancellation among terms), sorting before summation or using a priority queue can help without paying the full float64 cost.
