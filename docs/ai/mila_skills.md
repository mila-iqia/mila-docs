---
title: Install Mila Skills for Claude Code
description: >-
  Add the Mila skills marketplace to Claude Code and install cluster-focused
  skills for account setup, SSH connections, and job submission.
biel_boost: 0.5
---

# Install Mila Skills for Claude Code

The [mila-iqai/skills](https://github.com/mila-iqia/skills) marketplace is a
curated collection of Claude Code skills for Mila cluster workflows. Once
installed, each skill becomes available as a slash command that guides
researchers through common tasks — from setting up a Mila account to submitting
batch jobs with sbatch. This guide covers adding the marketplace to Claude Code
and installing the mila-* skills.

## Before you begin

<div class="grid cards" markdown>

-   [:material-robot:{ .lg .middle } __Set Up Claude Code__](claude_code.md)
    { .card }

    ---
    Install Claude Code and extend Claude Code with skills from a marketplace.

&nbsp;

</div>

## What this guide covers

* Add the [mila-iqai/skills](https://github.com/mila-iqia/skills) marketplace as
  a source in Claude Code
* Review the available Mila Skills
* Install and invoke a Mila Skill

---

## Add the Mila Skills marketplace

In a Claude Code session, run the following command to register the
mila-iqai/skills repository as a marketplace source:

```
/plugin marketplace add mila-iqia/skills
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin marketplace add mila-iqia/skills
  ⎿  Successfully added marketplace: mila-skills
```
</div>

Claude Code fetches the marketplace metadata and makes the available skills
visible for installation.

!!! note
    Verify the command above against the [mila-iqai/skills
    README](https://github.com/mila-iqia/skills) — the marketplace API may
    evolve over time.

## Available Mila Skills

The marketplace provides the following skills for Mila cluster workflows:

| Skill | Description |
|---|---|
| `/mila-account-setup` | Obtain a Mila account, enable cluster access, and configure multi-factor authentication. |
| `/mila-local-setup` | Install WSL2, uv, and milatools on a local machine. |
| `/mila-connect-cluster` | Connect to the cluster over SSH, enter a one-time password, and troubleshoot connectivity issues. |
| `/mila-run-jobs` | Run interactive development sessions and submit batch jobs with sbatch. |

## Install and invoke a skill

Install all Mila Skills with:

```
/plugin install mila-tools@mila-skills
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /plugin install mila-tools@mila-skills
  ⎿  ✓ Installed mila-tools. Run /reload-plugins to apply.
```
</div>

After installing, activate the skills in the current session without
restarting:

```
/reload-plugins
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ /reload-plugins
  ⎿  Reloaded: 1 plugin · 0 skills · 5 agents · 0 hooks · 0 plugin MCP servers · 0 plugin LSP servers
```
</div>

Once installed, invoke any skill as a slash command in a Claude Code session to
be guided through the selected workflow step by step or ask any related
questions:

```
I can't seem to login to the 2FA thing, I use my username and the supplied password in the email but it won't let me logon. What could be the reason for this?
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
 ▐▛███▜▌   Claude Code v2.1.109
▝▜█████▛▘  Sonnet 4.6 · Claude Team
  ▘▘ ▝▝    ~/CODE/project

❯ I can't seem to login to the 2FA thing, I use my username and the supplied
  password in the email but it won't let me logon. What could be the reason for
  this?

⏺ Skill(mila-tools:mila-account-setup)
  ⎿  Successfully loaded skill

⏺ Fetch(https://docs.mila.quebec/getting_started/)
  ⎿  Received 115.7KB (200 OK)

⏺ The most likely reason you can't log in is that you're using the wrong
  credentials. The MFA portal at https://mfa.mila.quebec uses different
  credentials from what you might expect:

  - Username: Your cluster username (not your @mila.quebec email address)
  - Password: The one-time registration token from the IT email with subject
    "Votre accès temporaire registrationcode / Your temporary access
    registrationcode" — not your Mila account password

  A few common scenarios:

  1. Wrong username — make sure you're using your cluster username, not your
     email address.
  2. Wrong password — the password field expects the registration token from
     that specific IT email, not your regular password.
  3. Token already used — if you previously logged in to the portal but left
     without adding a TOTP token, the registration token is now expired. You'd
     need to contact IT support at https://it-support.mila.quebec to get a new
     one.

  If you're still stuck after trying the above, reach out to IT support and ask
  for a new registration token.

  Sources:
  - https://docs.mila.quebec/getting_started/
```
</div>
