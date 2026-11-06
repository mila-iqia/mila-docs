---
title: Help us help you
description: >-
  Share experiment code you think could run better, and help IDT build and
  test tools that make everyone's jobs faster.
---

# Help us help you

In an research environment such as Mila, it is important to us to be transparent and reachable because we trust the force of community.
As we are currently not researchers (except for one member), former researchers at most, we do our best to reach out to researchers and understand their daily life and struggles.
We learn, but we also guess and conclude. In general, don't hesitate to reach us in your favorite communication way (mail, Slack, Git), every feedback or advice is more than appreciated!

More specifically, we are working on some projects requiring real-life testing and data to be more effective: like any tool built on real cases, it is only as
good as the real cases it was built from.

**That's where you come in.**

## Projects

| Project | Quick overview | What we need |
| ------- | -------------- | ------------ |
| [Optimization loop project](../ai/optimization.md) | Optimize code through agent Skills | Research experiment/code base |

### Optimization loop project

#### Description

When we profile someone's code at the [Office Hours](office_hours.md) or during
a compute-waste intervention, the same handful of problems keeps coming back: a
dataloader with too few workers, experiments small enough that a whole GPU
twice as large as they need, checkpoints written after every batch. Fixing them
is usually easy — *finding* them is the hard part, and it takes experience.

We are turning that experience into tools —
[skills your AI agent can use](../ai/optimization.md) to diagnose and fix these
problems automatically.

#### What we're asking for

If you look at your experiment code and think *"this probably runs worse than
it should"*, or simply *"I have no idea how fast this is supposed to be"* —
that's exactly the code we want.

We're especially interested in:

* **Workloads that feel slow** but where the bottleneck isn't obvious
* **Code that got "big"** — it started as one script and became a training
  suite across several clusters
* **Workload types that are under-represented** in our examples: RL loops,
  inference serving, heavy data preprocessing, multi-agent setups
* **Code that already runs well** — we need those too, as reference cases, so
  our tools learn what "good" looks like and stop "optimizing" what doesn't
  need it.

!!! tip "Perfection is not mandatory"

    You don't need working benchmark scripts, clean code, or a bug-free repo. Messy
    research code is the norm.

#### What happens to your code

Respectfully, and with your permission at every step:

* **You decide what you share.** That can be read access to a repo or a copy of
  the training loop and launch scripts without any data.
* **Your code stays internal.** Examples are kept privately. Your approval will
  be requested if we wish to test an optimization with non-local agents.
* **You get the findings back.** Every case we profile comes with what we
  found. Worst case, you get a free profiling session!

## How to share

<div class="grid cards" markdown>

-   :material-door-open:{ .lg .middle } **Come to the Office Hours**

    ---

    Bring your repo, we'll profile it together, and if it's a good candidate
    we'll ask to keep it (with your go-ahead).

    [:octicons-arrow-right-24: Office Hours](office_hours.md)

-   :fontawesome-brands-slack:{ .lg .middle } **Ping IDT on Slack**

    ---

    Say hello on TBD and we'll set up
    something that works for you — usually read access to a repo.

    [:octicons-arrow-right-24: #TBD](https://mila-umontreal.slack.com/archives/TBD)

-   :material-robot:{ .lg .middle } **Optimize first, share the report**

    ---

    Run the [optimization skill](../ai/optimization.md) with your own agent and
    send us the report it writes — it tells us where the tool struggled. It won't provide us your code, but it certainly helps us!

    [:octicons-arrow-right-24: Optimization skill](../ai/optimization.md)

&nbsp;

</div>

Even a *"yes, you can look at my repo"* is a complete answer. Every codebase we
collect makes the tools sharper for the next researcher.
