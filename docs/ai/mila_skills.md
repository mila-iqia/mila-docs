---
title: Mila Skills
description: >-
  Description of the cluster-focused Mila skills for account setup, SSH connections, and job submission.
---

# Mila Skills

The [mila-iqia/skills](https://github.com/mila-iqia/skills) repository is a
curated collection of Skills for Mila cluster workflows. Once
installed, each skill can be invoked by researchers through common tasks — from setting up a Mila account to submitting batch jobs with sbatch. This page describes the Skills.

## Before you begin

!!! note
    The Mila Skills are public and work with any harness —
    you can use one of this guide, but it is a recommendation, not a requirement.

<div class="grid cards" markdown>

-   [:material-robot:{ .lg .middle } __Set Up the Mila agent__](../getting_started/mila_local_agent.md)
    { .card }

    ---
    Install the Mila local agent and OpenCode. These Skills are in the
    provided configuration file.

-   [:material-robot:{ .lg .middle } __Set Up Skills with Claude Code__](claude_code.md#add-skills-and-plugins)
    { .card }

    ---
    Install Claude Code and extend Claude Code with skills from a marketplace.

&nbsp;

</div>

## What this guide covers

* Describe what a Skill is
* List the available Mila Skills
* Create your own Skill

---

## What is a Skill

**A Skill** is a reusable, self-contained guide that teaches an agent how to carry
out a specific task. A Skill adds **know-how**, not
**capability**: the model keeps using the tools the harness already provides,
and the Skill tells it how to combine them, in which order, and which Mila
specifics to respect.

**Physically, a Skill is a folder of files installed on the machine. Each folder
holds a `SKILL.md` file with two parts:

* a header naming the skill and describing when it applies — this is what
  turns the skill into a slash command such as `/mila-run-jobs`;
* a body of step-by-step instructions the model follows once the skill is
  used.

??? info "More details"
    Neither the model nor the harness contains the Skill: the instructions live on
    disk, and the harness decides which pieces of them the model gets to see, and
    when.

    1. **At startup** — the harness scans the installed Skill folders and places
      only the name and description of each Skill into the model's context. The
      model knows which Skills exist without reading any of them, so many Skills
      can stay installed at a small context cost.
    2. **On invocation** — the model recognises that the current task matches a
      description, or the researcher types the skill's slash command.
    3. **On demand** — the harness reads the full `SKILL.md` and adds the
      instructions to the context. The model follows them with its existing
      tools. The full guide enters the context only at this step — never before.

    <div class="wf2-loop">
      <svg viewBox="0 0 680 440" role="img" aria-labelledby="sk-flow-title" xmlns="http://www.w3.org/2000/svg">
        <title id="sk-flow-title">Skills live as plain files on disk. The harness reads those files and puts only each skill's name and description into the model's context. When a skill is invoked — the task matches a description, or a researcher types its slash command — the harness loads the full instructions into the context. Skills add know-how, not new tools.</title>
        <defs>
          <marker id="sk-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path class="lp-head" d="M0 0 L10 5 L0 10 z"/>
          </marker>
        </defs>
        <!-- Model band -->
        <rect class="lp-node lp-model" x="20" y="20" width="640" height="130" rx="10"/>
        <text class="lp-title" x="36" y="48">Model (LLM) — everything it can see right now</text>
        <rect class="lp-badge-model" x="572" y="32" width="72" height="18" rx="9"/>
        <text class="lp-badge-text lp-badge-model-text" x="608" y="45" text-anchor="middle">Context</text>
        <rect class="lp-node lp-model" x="40" y="68" width="280" height="58" rx="8"/>
        <text class="lp-io-text" x="56" y="90">Skill catalogue</text>
        <text class="lp-sub" x="56" y="110">name + description — always in context</text>
        <rect class="lp-io" x="345" y="68" width="300" height="58" rx="8"/>
        <text class="lp-io-text" x="361" y="90">Full instructions</text>
        <text class="lp-sub" x="361" y="110">read from disk only while a skill is used</text>
        <!-- Harness to model -->
        <path class="lp-edge" d="M140 206 V158" marker-end="url(#sk-flow-arrow)"/>
        <text class="lp-label" x="152" y="172">②</text>
        <path class="lp-edge" d="M480 154 V202" marker-end="url(#sk-flow-arrow)"/>
        <text class="lp-label" x="468" y="188" text-anchor="end">③</text>
        <path class="lp-edge lp-loop" d="M580 206 V158" marker-end="url(#sk-flow-arrow)"/>
        <text class="lp-label" x="568" y="172" text-anchor="end">④</text>
        <!-- Harness band -->
        <rect class="lp-node lp-harness" x="20" y="210" width="640" height="60" rx="10"/>
        <text class="lp-title" x="36" y="247">Harness</text>
        <text class="lp-sub" x="110" y="247">the program around the model — it reads the files and feeds them into the context</text>
        <!-- Disk to harness -->
        <path class="lp-edge" d="M340 328 V278" marker-end="url(#sk-flow-arrow)"/>
        <text class="lp-label" x="356" y="308">①</text>
        <!-- Disk band -->
        <rect class="lp-io" x="20" y="330" width="640" height="70" rx="10"/>
        <text class="lp-io-text" x="36" y="354">Skills on disk — plain files</text>
        <text class="lp-sub" x="36" y="376">mila-run-jobs/SKILL.md — a name + description header, then step-by-step instructions</text>
        <!-- Take-away -->
        <text class="lp-center" x="340" y="426" text-anchor="middle">Skills add know-how, not new tools — the model acts with the tools the harness already provides.</text>
      </svg>

      * ① the harness reads the installed Skill folders
      * ② the catalogue stays in context
      * ③ a skill is invoked: the task matches a description or a researcher types /mila-run-jobs
      * ④ the full guide is loaded — on demand
    </div>

## Available Mila Skills

The Skills provided Mila are gathered in the [Mila Skills Github repository](https://github.com/mila-iqia/skills).
They are divided into two plugins:

### `mila-tools` plugin

| Skill | Description |
|---|---|
| `/mila-account-setup` | Obtain a Mila account, enable cluster access, and configure multi-factor authentication. |
| `/mila-local-setup` | Install WSL2, uv, and milatools on a local machine. |
| `/mila-connect-cluster` | Connect to the cluster over SSH, enter a one-time password, and troubleshoot connectivity issues. |
| `/mila-run-jobs` | Run interactive development sessions and submit batch jobs with sbatch. |
| `/mila-paperoni` | Query and manage the Mila Paperoni publication database via its REST API at https://paperoni.mila.quebec. |
| `/mila-sarc` | Fetch job, user, cluster and usage data over HTTP from the SARC API at sarc.mila.quebec. |

### `opt-tools` plugin

| Skill | Description |
|---|---|
| [`/optimization-target`](optimization.md) | Optimize experiment code and compute usage on Slurm clusters so that many experiments finish as quickly as possible without changing or compromising their results. |

<!-- Steven's skills
| `/pytorch-compile` | Diagnose and fix torch.compile problems: graph breaks, recompilation, slow
  compile time, and compile-mode / runtime tradeoffs. |
| `/pytorch-dataloader` | Diagnose and fix PyTorch input pipelines that starve the GPU. |
| `/pytorch-profiler-full` | Find where a PyTorch training step's time goes with torch.profiler. |
| `/vibecheck` | Measurable, prioritized cleanup of a Python ML research repo before open-sourcing it. |
-->

## Create your own Skill

A Skill needs no registration or build step: a folder holding a `SKILL.md`
file, installed where the harness scans for skills, is complete. The
conventional layout groups the supporting material beside the guide:

```text title="mila-run-jobs/"
mila-run-jobs/
├── SKILL.md          # required — header + step-by-step instructions
├── references/       # optional — guides read only when needed
└── scripts/          # optional — helper scripts the instructions call
```

`SKILL.md` itself holds the two parts described in
[What is a Skill](#what-is-a-skill):

* a YAML header with `name` and `description`. The description states when
  the Skill applies; it is the only part permanently visible to the model,
  so it decides whether the model recognises and loads the Skill;
* a Markdown body of ordered instructions, with the Mila specifics — paths,
  flags, commands — spelled out rather than implied.

Paths written inside the body are relative to the folder containing
`SKILL.md`, so supporting files travel with the Skill. The body loads only
when the Skill is used, and supporting files only when the instructions refer
to them — a large reference costs nothing until the model actually opens it.

Install the folder where the harness discovers skills:

=== "OpenCode"

    * **All sessions** — `~/.config/opencode/skills/<skill-id>/SKILL.md`
    * **One project** — `.opencode/skills/<skill-id>/SKILL.md`

    [Add Skills and plugins](local_agent.md#add-skills-and-plugins) covers
    the other sources OpenCode accepts, such as shared catalogs.

=== "Claude Code"

    * **All sessions** — `~/.claude/skills/<skill-name>/SKILL.md`
    * **One project** — `.claude/skills/<skill-name>/SKILL.md`

    [Add Skills and plugins](claude_code.md#add-skills-and-plugins) covers
    installing skills from a marketplace instead.

!!! tip "Write the description first"
    The body can be refined over time; the description decides whether the
    Skill is ever loaded. State the situations where the Skill applies —
    "Use when submitting sbatch jobs on the Mila cluster" — rather than a
    summary of its contents.

---

## Next step

<div class="grid cards" markdown>

-   [:material-rocket-launch:{ .lg .middle } __Optimize Your Experiments with AI__](optimization.md)
    { .card }

    ---
    Discover the IDT optimization skill to identify bottlenecks, implement fixes, and speed up your experiments by leveraging your own coding agent.

&nbsp;

</div>
