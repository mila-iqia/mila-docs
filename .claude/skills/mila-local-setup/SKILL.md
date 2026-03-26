---
name: mila-local-setup
description: >-
  Use this skill when the user asks about setting up their local machine to
  connect to the Mila cluster. Trigger phrases include: "How do I install
  WSL", "How do I install uv", "How do I install milatools", "How do I
  configure milatools", "How do I set up my laptop for the cluster",
  "mila init", "uv tool install milatools", "curl uv install",
  "How do I install the mila CLI", "Windows setup for the cluster",
  "What tools do I need on my laptop".
version: 1.0.0
argument-hint: <wsl|uv|milatools>
---

# Local Machine Setup for the Mila Cluster

This skill guides users through installing and configuring the tools needed
on their local machine before connecting to the cluster: WSL (Windows only),
`uv`, and `milatools`.

## Reference documentation

Primary source: **https://docs.mila.quebec/Userguide_quick_start/**
— sections "I'm using Windows, how do I install WSL?", "Install uv on your
local machine", "Install milatools", and "Configure milatools".

## Workflow

### Step 1: Identify the user's OS

Ask or infer the user's operating system:

- **Windows** → guide through WSL first, then uv and milatools inside WSL.
- **macOS or Linux** → skip WSL, go directly to uv and milatools.

If not mentioned, ask: "Are you on Windows, macOS, or Linux?"

### Step 2: Fetch the documentation

Fetch **https://docs.mila.quebec/Userguide_quick_start/** and locate the
sections relevant to the user's OS and question.

### Step 3: Guide through the setup steps in order

Work through the applicable steps:

**WSL (Windows users only):**
1. Open PowerShell and run `wsl --install Ubuntu`.
2. Restart the computer when prompted.
3. After restart, WSL finishes setup; create a Linux username and password
   if prompted.
4. Open Ubuntu from the Start menu to get a Linux terminal.
5. Verify: run `ls` and `curl --version` in the WSL terminal.
6. All subsequent commands (`uv`, `milatools`, `ssh`) must be run inside
   the WSL terminal, not PowerShell or Command Prompt.

**Install `uv` (all platforms, inside WSL on Windows):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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
