---
title: Optimize your experiments with AI
description: >-
  Pair your coding agent with a Mila skill to diagnose bottlenecks, optimize your workload, and achieve faster results.
---

# Optimize your experiments with AI

If you are facing slow experiments, constrained compute resources, or low utilization, there is a good chance your agent, equipped with adequate guidance and context, can help you out.

Coding agents already have broad knowledge of PyTorch and Slurm; what they lack is
Mila-specific context and specific optimization methods. IDT packages that
context into a **skill** the agent can load in order to:

1. **Diagnose** — measure actual job behavior (throughput, resources utilization)
   instead of guessing
2. **Recommend** — propose concrete fixes (dataloader workers, `torch.compile`,
   job packing, checkpointing frequency…) ranked by expected gain
3. **Fix safely** — implement the changes while keeping experimental results
   identical, and record the findings in a report.

You're always welcome for the [IDT Office Hours](../help/office_hours.md) to get help on this.

!!! wip "Work in progress"
    This is an early version: coverage of real-world cases is still being
    established.

    It explicitely requests confirmation to:
    * apply changes that could alter experimental results
    * share reports to the IDT team so that the skill may benefit from a feedback loop.

## The optimization skill

The skill lives in the [mila-iqia/skills][skills] marketplace, as the
`opt-tools` plugin.  To use it to optimize your code, start a session with your agent at the root of the experiment code and
    describe the desired outcome in plain language:

=== "Use the Skill (all agents)"

    The request to use the Skill can also be added directly in the prompt:
    ```
    Try to optimize my code using guidance from https://raw.githubusercontent.com/mila-iqia/skills/refs/heads/master/opt-tools/skills/optimization-target/SKILL.md.
    ```

=== "Use the Skill with Claude Code"

    If the marketplace was registered following
    [Install Mila Skills for Claude Code](mila_skills.md), only the install step
    is new:

    ```
    /plugin install opt-tools@mila-skills
    ```

    Claude Code can then be called to optimize the code

    ```
    Try to optimize my code.
    ```   

The agent detects the skill and follows its requirements:

* **Maximize resources utilization:** The core objective of the skill, achieved through diagnoses and standard improvement strategies like job packing.
* **Protect the science:** Optimization must preserve result integrity. The skill explicitly requires user confirmation before making any changes that could alter findings (such as hyperparameter adjustments).
* **Improve the Skills:** An optimization report is generated and can be submitted to the relevant repository to document identified found issues, applied fixes and performance gains.

!!! info "Sharing the report is optional"
    The optimization report is written locally and summarizes the experiment's purpose, its stack, its identified issues and the applied fixes.

    **Sharing it remains the user's choice**

    * The user can review and update it before submission.
    * It is possible to simply ask the agent not to share the report.
    * The report submission needs a drop directory name (the agent may refer to it as `DROPDIR`). Not providing the directory name will lead to the agent not sharing the report.

## Feedback

Feedback on real-world use drives the next iterations:

* Did the agent find real issues, or suggest changes with no effect?
* Did it miss something a human reviewer would catch immediately?
* Was the measurement sound — did the results stay the same?

Reach IDT on Slack ([#mila-cluster][slack] or [#compute-canada][slack-cc]),
come to the [Office Hours](../help/office_hours.md), or have the agent include
its observations in the report it sends (the skill asks it to). Negative
experiences are as valuable as positive ones.

Code with optimization potential can also be shared directly with IDT, as
described in [Help us help you](../help/feedback.md): real codebases are what
these tools are trained and improved on.

[skills]: https://github.com/mila-iqia/skills
[slack]: https://mila-umontreal.slack.com/archives/CFAS8455H
[slack-cc]: https://mila-umontreal.slack.com/archives/C2TL9FRBP

## Related pages

<div class="grid cards" markdown>

-   [:material-toy-brick:{ .lg .middle } __Install Mila Skills for Claude Code__](mila_skills.md)
    { .card }

    ---
    Add the Mila skills marketplace to Claude Code.

-   [:material-chart-line:{ .lg .middle } __Identifying GPU waste__](../userguides/compute_utilization_guidelines/profiling.md)
    { .card }

    ---
    Understand the utilization metrics behind the waste notifications.

-   [:material-account-question:{ .lg .middle } __Help us help you__](../help/feedback.md)
    { .card }

    ---
    Share code with optimization potential, and help build better tools.

&nbsp;

</div>
