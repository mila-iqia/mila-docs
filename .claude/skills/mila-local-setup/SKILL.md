---
name: mila-local-setup
description: >-
  Use this skill when the user asks about setting up their local machine to
  connect to the Mila cluster. Trigger phrases include: "How do I install
  WSL", "How do I install uv", "How do I install milatools", "What is
  milatools", "How do I configure milatools", "How do I set up my laptop for
  the cluster", "mila init", "uv tool install milatools", "curl uv install",
  "How do I install the mila CLI", "Windows setup for the cluster",
  "What tools do I need on my laptop", "SSH config", "SSH config setup",
  "authorized_keys".
version: 1.0.0
argument-hint: <wsl|uv|milatools>
---

# Local Machine Setup for the Mila Cluster

This skill guides users through installing and configuring the tools needed
on their local machine before connecting to the cluster: WSL (Windows only),
`uv`, and `milatools`.

## Base policies

At the start of each response, use the Read tool to load
`.claude/skills/mila-base/SKILL.md` and apply all policies defined there
before proceeding with the workflow below.

## Reference documentation

Primary source: **https://docs.mila.quebec/getting_started/index**
— sections "I'm using Windows, how do I install WSL?", "Install uv on your
local machine", "Install milatools", and "Configure milatools".

## Discover documentation

Use the WebSearch tool with this query to find the current URL of the primary
source above:

    site:docs.mila.quebec "__skill-mila-local-setup"

Use the URL from the search result in the WebFetch steps below. If the search
returns no results, fall back to the hardcoded URL in "Reference documentation".

## Workflow

### Step 1: Identify the user's OS

Ask or infer the user's operating system:

- **Windows** → guide through WSL first, then uv and milatools inside WSL.
- **macOS or Linux** → skip WSL, go directly to uv and milatools.

If not mentioned, ask: "Are you on Windows, macOS, or Linux?"

### Step 2: Fetch the documentation

Use the WebFetch tool to fetch **https://docs.mila.quebec/getting_started/index** and locate the
sections relevant to the user's OS and question.

### Step 3: Guide through the setup steps in order

Work through the applicable steps:

**WSL 2 (Windows users only):** WSL 2 is required (not WSL 1); the install
command below installs WSL 2 by default on Windows 10 21H2 and later, and
on all versions of Windows 11.

1. Open PowerShell and run `wsl --install Ubuntu`.
2. Restart the computer when prompted.
3. After restart, verify WSL 2 is active: in PowerShell, run
   `wsl --list --verbose`. The VERSION column should show `2` for Ubuntu.
   If it shows `1`, run `wsl --set-version Ubuntu 2` to upgrade.
4. WSL finishes setup; create a Linux username and password if prompted.
5. Open Ubuntu from the Start menu to get a Linux terminal.
6. Verify: run `ls` and `curl --version` in the WSL terminal.
7. All subsequent commands (`uv`, `milatools`, `ssh`) must be run inside
   the WSL terminal, not PowerShell or Command Prompt.

**Install `uv` (all platforms, inside WSL on Windows):**
Use the WebFetch tool to fetch **https://docs.astral.sh/uv/getting-started/installation/** for
up-to-date installation instructions. Follow the steps for the user's
platform (Linux/WSL, macOS, or Windows).

**Install `milatools`:**
```bash
uv tool install --upgrade milatools
```

**Configure `milatools` with `mila init`:**
- Have the cluster username ready (received from IT after passing the
  onboarding quiz).
- MFA must already be set up before running `mila init` (see
  **mila-account-setup**).
- Run `mila init` and follow the prompts: it sets up the SSH config,
  generates SSH keys, and copies the public key to the cluster's
  `~/.ssh/authorized_keys`.

```bash
mila init
```

### Step 4: Answer follow-up questions

Common questions:

- "What is `uv`?" — A fast Python package manager and workflow tool,
  used both locally and on the cluster. It installs `milatools` and later
  manages Python environments for cluster jobs.
- "Do I need to run `mila init` again if I reinstall?" — Only if the SSH
  config or keys are missing. Re-running `mila init` is safe.
- "How do I update `milatools`?" — `uv tool install --upgrade milatools`
  (same install command).
- "Can I use Cursor instead of VSCode?" — Yes, `mila code` supports
  Cursor and other compatible editors. See **mila-run-jobs** for details.

### Step 5: Point to the next skill

Once tools are installed and `mila init` has completed successfully, point
the user to **mila-connect-cluster** to verify the SSH connection.
