---
title: Set Up Claude Code
description: >-
  Install Claude Code and extend Claude Code with skills from a marketplace.
---

# Set Up Claude Code

Claude Code is Anthropic's official command-line interface for Claude.
It brings an AI coding assistant directly into the terminal, desktop, or
IDE, enabling researchers to generate code, explain errors, and automate
repetitive tasks without leaving their development environment. This guide
covers installation, authentication, and skills — reusable extensions that
add domain-specific behavior to Claude Code.

## Before you begin

!!! success "Requirements"
    - An [Anthropic account](https://claude.ai) with an active Claude Pro,
      Team, or Enterprise subscription.
    - macOS, Linux, or [Windows with WSL2
      configured](../getting_started/index.md#install-wsl) (for CLI
      installation).

## What this guide covers

* Install Claude Code using the CLI installer, desktop app,
  or IDE extension
* Authenticate with an Anthropic account
* Install skills from a skills marketplace

---

## Install Claude Code

Claude Code is available as a command-line tool and as an extension for VS Code.

=== "CLI (macOS / Linux)"

    Run the installer in a terminal:

    ```bash
    curl -fsSL https://claude.ai/install.sh | bash
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    Setting up Claude Code...

    ✔ Claude Code successfully installed!

      Version: 2.1.109

      Location: ~/.local/bin/claude


      Next: Run claude --help to get started

    ✅ Installation complete!
    ```
    </div>

    Verify the installation:

    ```bash
    claude --version
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    2.1.109 (Claude Code)
    ```
    </div>

=== "CLI (Windows — WSL2)"

    ???+ warning "WSL2 required"
        The CLI installer requires a Linux environment. Install and configure
        WSL2 before continuing. See [Getting Started — Install
        WSL](../getting_started/index.md#install-wsl).

    Open a WSL2 terminal and run the installer:

    ```bash
    curl -fsSL https://claude.ai/install.sh | bash
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    Setting up Claude Code...

    ✔ Claude Code successfully installed!

      Version: 2.1.109

      Location: ~/.local/bin/claude


      Next: Run claude --help to get started

    ✅ Installation complete!
    ```
    </div>

    Verify the installation:

    ```bash
    claude --version
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    2.1.109 (Claude Code)
    ```
    </div>

=== "IDE extension"

    Install the Claude Code extension from the marketplace for the IDE:

    - **VS Code** — search for "Claude Code" in the [VS Code
      Marketplace](https://marketplace.visualstudio.com/)

!!! note
    For the most up-to-date installation instructions, see the [Claude Code
    product page](https://claude.com/product/claude-code).

## Authenticate

After installation, launch Claude Code to complete authentication.

For the CLI, open a terminal and run:

```bash
claude
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

--------------------------------------------------------------------------------
  Login


   Claude Code can be used with your Claude subscription or billed based on API
   usage through your Console account.

   Select login method:

   ❯1. Claude account with subscription · Pro, Max, Team, or Enterprise

    2. Anthropic Console account · API usage billing

    3. 3rd-party platform · Amazon Bedrock, Microsoft Foundry, or Vertex AI


  Esc to cancel
```
</div>

Claude Code opens a browser window. Complete the Anthropic login flow in the
browser; the terminal session authenticates automatically when login succeeds.
The desktop app and IDE extensions display a built-in login prompt on first
launch.

## Install skills from a marketplace

Skills extend Claude Code with reusable, domain-specific behavior. Each skill is
a self-contained workflow invoked as a slash command — for example,
`/code-review` or `/skill-creator`. Skills are distributed through
GitHub-hosted marketplaces and can be installed into any Claude Code session.

### Add a marketplace

Skills marketplaces are GitHub repositories containing skill definitions.
Register a marketplace as a source before installing its skills:

```
/plugin marketplace add <github_owner/repo>
```

!!! note
    The official Anthropic marketplace (`claude-plugins-official`) is
    pre-registered and requires no `add` step.

### Install a skill

Once a marketplace is registered, install skills by name using the format
`skill-name@marketplace-name`:

```
/plugin install code-review@claude-plugins-official
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin install code-review@claude-plugins-official
  ⎿  ✓ Installed code-review. Run /reload-plugins to apply.
```
</div>

After installing, activate the skill in the current session without
restarting:

```
/reload-plugins
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin install code-review@claude-plugins-official
  ⎿  ✓ Installed code-review. Run /reload-plugins to apply.

❯ /reload-plugins
  ⎿  Reloaded: 1 plugins · 0 skill · 5 agents · 0 hooks · 0 plugin MCP servers · 0 plugin LSP servers
```
</div>

Run `/plugin` and navigate to the **Installed** tab to list all currently
installed skills.

```
/plugin
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin

--------------------------------------------------------------------------------
  Plugins  Discover  *Installed*  Marketplaces   Errors

  ╭────────────────────────────────────────────────────────────────────────────╮
  │ ⌕ Search…                                                                  │
  ╰────────────────────────────────────────────────────────────────────────────╯

    Local
  ❯ code-review Plugin · claude-plugins-official · ✔ enabled

   type to search · Space to toggle · Enter to details · Esc to back
```
</div>

### Explore the official skills library

The `/plugin` command opens an interactive manager with four tabs: **Discover**,
**Installed**, **Marketplaces**, and **Errors**. Use the **Discover** tab to
browse skills across registered marketplaces.

The official Anthropic marketplace includes skills in four categories:

- **Code intelligence** — language server integrations for Python, Rust,
  TypeScript, Go, and more
- **External integrations** — GitHub, GitLab, Jira, Figma, Slack, Vercel, and
  other services
- **Development workflows** — commit tools, PR review, and plugin development
- **Output styles** — response customization

For the full catalogue, see the [Claude Code plugin
documentation](https://code.claude.com/docs/en/discover-plugins).

### Manage installed skills

#### Update a marketplace

Pull the latest skill definitions from a registered marketplace:

```
/plugin marketplace update claude-plugins-official
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin marketplace update claude-plugins-official
  ⎿  ✔ Updated 1 marketplace
```
</div>

#### Remove a skill

Uninstall a skill from the current user scope:

```
/plugin uninstall code-review
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin uninstall code-review
  ⎿  ✓ Enabled code-review. Run /reload-plugins to apply.
```
</div>

#### Remove a marketplace source

Remove a marketplace and make its skills unavailable for future installs:

```
/plugin marketplace remove github_username/project
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin marketplace remove mila-skills
  ⎿  ✔ Removed 1 marketplace
```
</div>

---

## Key concepts

**Claude Code**
:   Anthropic's official CLI and IDE extension for Claude. Provides an AI coding
    assistant in the terminal or IDE, extendable with skills.

**Skill**
:   A self-contained workflow that extends Claude Code with domain-specific
    behavior. Skills are invoked as slash commands (e.g.
    `/mila-connect-cluster`) in a Claude Code session.

**Marketplace**
:   A GitHub repository containing one or more skill definitions. Adding a
    marketplace as a source makes its skills available to install.

## Next step

<div class="grid cards" markdown>

-   [:material-toy-brick:{ .lg .middle } __Install Mila Skills for Claude Code__](mila_skills.md)
    { .card }

    ---
    Add the Mila skills marketplace and install cluster-focused skills for
    account setup, SSH connections, and job submission.

&nbsp;

</div>
