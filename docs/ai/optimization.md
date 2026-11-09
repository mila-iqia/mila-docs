---
title: Optimize your experiments with AI agents
description: >-
  IDT's optimization loop: use your coding agent with a Mila skill to
  diagnose and fix underused GPUs in your experiment code.
---

# Optimize your experiments with AI agents

Users periodically receive notifications that jobs use GPUs inefficiently.
Such notifications state that a problem exists, not what to change in the
code. Discussing a specific case with IDT at the
[Office Hours](../help/office_hours.md) is effective, but one-on-one support
does not scale across the community.

**IDT is working on closing this gap.** Coding agents (Claude Code or
similar) already have broad knowledge of PyTorch and Slurm; what they lack is
Mila-specific context and specific optimization methods. IDT packages that
context into a skill the agent can load in order to:

1. **Diagnose** — measure actual job behavior (throughput, resources utilization)
   instead of guessing
2. **Recommend** — propose concrete fixes (dataloader workers, `torch.compile`,
   job packing, checkpointing frequency…) ranked by expected gain
3. **Fix safely** — implement the changes while keeping experimental results
   identical, and record the findings in a report.

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

* **Maximize resources utilization:** It is the main goal or the Skill, and can be done through multiple vectors such as jobs packing.
* **Protect the science:** The optimization should have no impact on the results. Hyperparameters changes for instance can alter results, so the skill explicitely ask precisions to the user for such an operation.
* **Improve the Skills:** An `optimization-report.md`is generated and can be sent in the repository mentioned in Slack usage notifications. This reports describes the issues found, the fixes made, and the gains obtained.

## The report is optional

!!! info "Report directory"
    The `DROPDIR` needed to automatically send a report is mentionned in your Slack resources usage notification.

Once the work is done, the agent may offer to send the report to IDT. Reports
show which problems recur across the cluster and which optimizations transfer
from one codebase to another. **Sharing remains the user's choice.** The
report is written locally in the repository, and nothing leaves the machine
without explicit confirmation.

To keep the report local:

* tell the agent **not to upload the report** (for example: *"do not send the
  report to IDT, keep it local only"*), or
* do not provide the drop directory — the agent only uploads when it has been
  given one, and does not retry a failed upload.

The report is a summary written for an outside reader (what the code does, the
stack, the issues and the fixes). Before sharing it, feel free to review the file and remove
anything that should not be shared; the file remains under the author's
control.

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
