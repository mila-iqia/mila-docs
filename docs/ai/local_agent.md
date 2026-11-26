---
title: Use the Mila local agent
description: >-
  Daily-use guide for the Mila local agent: start the SSH tunnel, launch
  OpenCode, manage the global and project configuration, add Skills and
  plugins, and provide permanent or per-session context.
---

# Use the Mila local agent

Mila provides a **locally hosted** AI agent.

In this guide:

* **the model** is a LLM hosted on the Mila cluster
* **the harness** is [OpenCode](http://opencode.ai/), but feel free to use your own.

The [Get Started with the Mila Coding Agent](../getting_started/mila_local_agent.md)
covers the one-time setup on a local machine.

??? info "Overview of the agent"
    This open-weight language model (currently `Qwen3.8 Flash Next Fp8`) is
    served behind a LiteLLM gateway on a DGX-A100 node of the Mila cluster.

    The gateway exposes an OpenAI-compatible API, so any agent harness that
    supports such an endpoint can use the agent instead of a cloud provider
    — conversations, code, and data stay within the Mila ecosystem.

!!! warning "Beta"
    The Mila agent is in beta. Announcements about newly deployed models and
    changes are posted on the `#mila-local-agent` Slack channel.

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __First use of the Mila Coding Agent__](../getting_started/mila_local_agent.md)
    { .card }

    ---
    Get an API key, route traffic to the agent node, and
    configure OpenCode.

&nbsp;

</div>

## What this guide covers

* Start a working session with the agent (SSH tunnel and OpenCode)
* Global configuration and project configuration of OpenCode
* Add Skills and plugins to the harness
* Add context to the agent — permanently or for a single session

!!! warning "Installation is **not** covered here"
    See [Get Started with the Mila Coding Agent](../getting_started/mila_local_agent.md) for this.

---

## Start a working session

The agent node is reachable only through a Mila login node, so a working
session starts with an SSH tunnel. The `/etc/hosts` entry and the API key
added during the [first-use setup](../getting_started/mila_local_agent.md) are persistent — setting them up again is
not needed.

1. Open a terminal and start the SSH tunnel with port forwarding:

    ```bash
    ssh mila -L 8080:cn-d003.server.mila.quebec:8080
    ```

    Keep this terminal open for the whole work session — closing it drops the
    route to the agent node.

2. Move to the repository to work on and launch OpenCode:

    ```bash
    cd ~/code/my-project
    opencode
    ```

3. Verify the session is correctly connected:

    * **The model** is `Qwen3.8 Flash Next Fp8`.
    * **The repository** shown is the intended project.

??? tip "Check the connection without OpenCode"

    With the tunnel active, a direct request confirms that the gateway is
    reachable:

    ```bash
    curl "http://cn-d003.server.mila.quebec:8080/v1/chat/completions" \
      -H "Authorization: Bearer $MILA_AGENT_KEY" \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.8-Flash-Next-FP8","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16}'
    ```

    A successful call returns a JSON object whose message content echoes the
    prompt.

## Use of another harness

OpenCode is the recommended harness used in this guide, but it is not required.
With another harness, point it to the following OpenAI-compatible endpoint:

``` linenums="0"
http://cn-d003.server.mila.quebec:8080/v1/
```

## Configure OpenCode

??? tip "Configuration file"

    ```json title="~/.config/opencode/opencode.jsonc"
    --8<-- "docs/ai/opencode_configuration.json"
    ```

OpenCode reads two levels of configuration and merges them:

| Level | File | Applies to |
|---|---|---|
| **Global** | `~/.config/opencode/opencode.jsonc` | every project |
| **Project** | `opencode.jsonc` at the project root, or `.opencode/opencode.jsonc` | that project only |

=== "Global configuration"

    This file is created in the [getting-started guide](../getting_started/mila_local_agent.md#configure-opencode). It holds everything that should
    apply to every project:

    * the `a100` provider pointing at the agent gateway, with the API key read
    from `MILA_AGENT_KEY` at runtime,
    * the suggested guardrails (`permissions` and `policies` rules),
    * the `opencode-models-discovery` plugin,
    * the Mila Skills catalogs and the Mila Docs MCP server.

    Keep anything machine-wide or personal here — a preferred model, guardrails,
    or shared catalogs.

=== "Project configuration"

    Add `opencode.jsonc` at the root of a repository to configure that project
    alone: a default model, a formatter, project-specific skills directories, or
    MCP servers. Commit the file so everyone working in the repository gets the
    same setup.

!!! info "Files are merged, not replaced:"

    * Values in the project file override conflicting global values; settings
    that do not conflict are preserved.
    * List-type settings such as `skills` and plugins are combined across files
    rather than replaced.

## Add Skills and plugins

### Skills

Skills are reusable step-by-step guides the agent loads on demand — only the
name and description of each skill sit in the context; the full instructions
are loaded when the skill is actually used.

!!! tip "Mila Skills"
    Mila publishes curated Skills adapted to the lab workflows — see
    [Mila Skills](mila_skills.md).

There are different ways to add Skills with OpenCode: the **automatic discovery** and through the **configuration file**.

=== "Automatic discovery"

    OpenCode discovers skills automatically in:

    | Scope | Location |
    |---|---|
    | Global | `~/.config/opencode/skills` |
    | Project | `.opencode/skills` |

=== "Configuration file"
    Extra sources — local directories or HTTP catalogs — can be listed in the
    `skills` array of any configuration file. The Mila global configuration
    already loads the two Mila catalogs from the
    [mila-iqia/skills](https://github.com/mila-iqia/skills) repository.

??? info "Personal skills"
    To create a personal skill, add a directory with a `SKILL.md` file under
    `~/.config/opencode/skills`:

    ```markdown title="~/.config/opencode/skills/slurm-logs/SKILL.md"
    ---
    name: Slurm logs
    description: Extract errors and timings from Slurm job output files
    ---

    1. Locate the `slurm-<jobid>.out` file for the job.
    2. Grep for error markers and summarize the failure with the failing step.
    ```

    The `description` is what lets the model pick the skill at the right moment.
    To load a specific skill manually, mention it as `@skill-id` in the prompt.

### Plugins

Plugins are packages or local directories that extend the harness with tools,
hooks, or commands.

Such as the Skills, there are several ways to add plugins with OpenCode: an
**automatic discovery**, through the **configuration file** and with
**command line instructions**.

=== "Automatic discovery"
    Project-local plugins are discovered under `.opencode/plugins/`.

=== "Configuration file"
    Add an entry to the plugin list of any configuration
    file — the getting-started configuration loads `opencode-models-discovery@latest`
    this way, which is why newly deployed models appear in OpenCode without a
    configuration change.

=== "Command line instructions"
    Global plugins can also be managed from the command line:

    ```bash
    opencode plugin add opencode-models-discovery@latest
    opencode plugin list
    opencode plugin update
    ```

## Add context

Context is everything the agent knows at a given moment — instructions,
project files, notes. Add it at three levels of persistence.

=== "Permanent context"

    An `AGENTS.md` file gives the agent standing instructions, loaded at the
    start of every session:

    * **Global** — `~/.config/opencode/AGENTS.md`: personal preferences that
    apply everywhere (preferred language, code style, habits to avoid).
    * **Project** — `AGENTS.md` at the repository root: build commands,
    architecture notes, and conventions. Commit it so every collaborator's
    agent behaves the same. More specific `AGENTS.md` files can sit in
    subdirectories and apply to the code below them.

    Additional instruction files — plain Markdown, glob patterns, or URLs — can
    be listed in the `instructions` array of a configuration file. Where no
    `AGENTS.md` exists, OpenCode reads `CLAUDE.md` instead.

=== "Context for one session"

    * **File mentions** — type `@` in the prompt to search and attach a file; a
    line range works too: `@src/auth.ts#20-45`.
    * **Direct input** — paste snippets, error traces, or notes into the prompt.
    * **Shell mode** — type `!` at an empty prompt to run a command; its output
    stays in the session, so the agent sees it: `!squeue --me`.

=== "Context on demand"

    * **Skills** hold task-specific know-how and load only when relevant —
    without occupying the context otherwise.
    * **MCP servers** connect the agent to external services. The Mila
    configuration already includes the Mila Docs MCP server, letting the agent
    look up cluster documentation by itself — see
    [The Mila Docs MCP server](mcp.md) and
    [Add Mila Docs context to AI assistants](context.md).

---

## Key concepts

Global configuration
:   The user-level file (`~/.config/opencode/opencode.jsonc`) applied to
    every project, for providers, guardrails, plugins, skills, and MCP
    servers.

Project configuration
:   A repository-level `opencode.jsonc` merged on top of the global
    configuration, for settings specific to one project.

Skill
:   A reusable set of instructions (`SKILL.md`) for a specific task,
    advertised to the model by name and description, loaded in full only
    when used.

Plugin
:   A package or local directory extending the harness with tools, hooks,
    or commands.

## Next steps

<div class="grid cards" markdown>

-   [:material-toy-brick:{ .lg .middle } __Mila Skills__](mila_skills.md)
    { .card }

    ---
    Overview of the Mila Skills to automate common cluster workflows.

-   [:material-book:{ .lg .middle } __Learn from Mila shared knowledge__](context.md)
    { .card }

    ---
    Give an AI assistant access to the Mila documentation through the MCP
    server or a condensed Markdown file.

</div>
