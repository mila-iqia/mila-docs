---
title: Logging in to the cluster
description: >-
  Configure SSH, log in to the Mila login nodes and connect to compute nodes.
---

<!-- START -->

# Logging in to the cluster

This guide describes the advanced ways to connect to the Mila cluster:
configuring SSH automatically or manually, logging in to a login node,
completing the Multi-Factor Authentication (MFA) prompt and connecting to
compute nodes. For a quick and simple way to connect, follow the
[Getting started](../../getting_started/index.md) section instead.

[](){ #ssh-secure-shell }


## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Get started with the cluster__](../../getting_started/index.md)
    { .card }

    ---
    First connection: enable cluster access, set up MFA and connect.

</div>

## What this guide covers

* Configure SSH for the Mila login nodes, with `mila init` or manually
* Log in to a login node and complete the MFA prompt
* Use the login nodes responsibly
* Connect to compute nodes, directly or with `mila-cpu`
* Use a non-Bash Unix shell

---

## Configure SSH

SSH uses the configuration file `~/.ssh/config` (next to the SSH keys) to
determine which connection settings to use for each SSH server.

The Mila **login** nodes require:

- `Hostname`: `login.server.mila.quebec`
- `Port`: `2222`
- `User`: *your Mila account username*
- `PreferredAuthentications`: `publickey,keyboard-interactive`

[`mila init`](#mila-init) automatically adds a simple SSH configuration to
`~/.ssh/config`. Advanced users can alternatively edit `~/.ssh/config`
[manually](#manual-ssh-configuration).

[](){ #mila-init }

### Configure SSH with `mila init`

Mila publishes the [milatools](https://www.github.com/mila-iqia/milatools)
package to help set up a productive environment. Its `mila init` command
performs some of the steps below automatically. It requires Python 3.9 or
later:

```bash
pip install milatools
mila init
```

!!! note
    This guide is current for `milatools >= 0.0.17`. To upgrade an older
    installation, run `uv tool upgrade milatools` or
    `pip install -U milatools`, then re-run `mila init` to apply new features
    and bug fixes.

### Manual SSH configuration

For advanced use cases, use the following template as a starting point for
the entries in `~/.ssh/config`:

```sshconfig
#   Mila
Host mila             login.server.mila.quebec
    Hostname          login.server.mila.quebec
Host mila1    login-1.login.server.mila.quebec
    Hostname  login-1.login.server.mila.quebec
Host mila2    login-2.login.server.mila.quebec
    Hostname  login-2.login.server.mila.quebec
Host mila3    login-3.login.server.mila.quebec
    Hostname  login-3.login.server.mila.quebec
Host mila4    login-4.login.server.mila.quebec
    Hostname  login-4.login.server.mila.quebec
Host mila5    login-5.login.server.mila.quebec
    Hostname  login-5.login.server.mila.quebec
Host cn-????
    Hostname             %h.server.mila.quebec
Match host !*login.server.mila.quebec,*.server.mila.quebec
    Hostname                 %h
    ProxyJump                mila
Match host           *login.server.mila.quebec
    Port                     2222
    ServerAliveInterval      120
    ServerAliveCountMax      5
Match host *.server.mila.quebec
    PreferredAuthentications publickey,keyboard-interactive
    AddKeysToAgent           yes
    ## Consider uncommenting:
    # ForwardAgent             yes
    ## Delete if on Linux, uncomment if on Mac:
    # UseKeychain              yes
    User                     CHANGEME_YOUR_MILA_USERNAME
```

!!! warning "Edit the template"
    Make the required edits to the template above, especially replacing
    `CHANGEME_YOUR_MILA_USERNAME` with your Mila username.

## Log in to a login node

### Connect with SSH

Login to the Mila cluster goes through `ssh`, four Internet-facing
*login nodes* and a load-balancer. At each connection, the load-balancer
redirects the session to the least loaded login node.

```bash
# Generic login, will send you to one of the 4 login nodes to spread the load
ssh -p 2222 <user>@login.server.mila.quebec

# To connect to a specific login node, X in [1, 2, 3, 4]
ssh -p 2222 <user>@login-X.login.server.mila.quebec
```

To avoid typing the full command each time, add a `mila` alias to the SSH
configuration file (see [Configure SSH](#configure-ssh)). With SSH correctly
configured, connect to a login node with:

```bash
# Login with SSH configuration in place
ssh mila

# Can also scp...        vvvv
scp  file-to-upload.zip  mila:scratch/uploaded.zip

#          vvvv  ... and rsync!
rsync -avz mila:my/remote/sourcecode/  downloaded-source/
```

The `mila` alias is available to `ssh`, `scp`, `rsync` and all other programs
that read the SSH configuration file.

!!! tip "Run commands remotely"
    You can run commands on the login node with `ssh` directly, for example
    `ssh mila squeue -u '$USER'`. Put single quotes around any `$VARIABLE`
    to evaluate on the remote side; otherwise the local shell evaluates it
    before `ssh` even runs.

### First login

On the first login, SSH may ask for the SSH key passphrase. Enter the
passphrase chosen when
[generating the SSH key](../../technical_reference/general_theory/ssh_on_clusters.md#generating-an-ssh-private-key).

SSH may also ask whether to trust the ***Mila*** login servers' *own* SSH
keys. The ECDSA, RSA and ED25519 fingerprints for Mila's login nodes are:

```text
SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8 (ECDSA)
SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o (RSA)
SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg (ED25519)
```

!!! warning "Verify the fingerprints"
    If the presented fingerprints do not match one of the above, **do not**
    trust them!

### Complete the MFA prompt

When connecting, the Mila cluster requests a MFA confirmation. This is done through
the option chosen while [setting up the MFA](../../getting_started/mfa.md): TOTP code,
Push on phone or email.

!!! info "When Push and OTP are both enrolled"
    The terminal always displays an OTP prompt, regardless of which factor is
    used.

    - **To use TOTP or email:** enter the code and press `Enter`.
    - **To use Push:** approve the notification on the smartphone first, then
      press `Enter` without entering a code.

!!! tip "Reduce MFA prompts with SSH connection multiplexing"
    Adding `ControlMaster=auto` and `ControlPersist=yes` to the Mila
    SSH config entry allows MFA verification once per machine boot
    rather than once per SSH command. This is supported on Linux and
    macOS. Windows users should install WSL to use these SSH options.

## Use login nodes responsibly

!!! warning "Do not run VSCode remote SSH instances directly on login nodes"
    Even though they are typically not very computationally expensive, when many people do it, they add up. See [Visual Studio Code](../../toolbox/development/VSCode.md) for specific instructions.

Login nodes are merely *entry points* to the cluster. They give access to the
compute nodes and to the filesystem, but they are not meant to run anything
heavy. Do **not** run compute-heavy programs on these nodes: doing so could
bring them down and impede cluster access for everyone.

This means no training scripts or experiments, and no compilation of software
unless it is small or ends quickly. Do not run anything that demands a
sustained large amount of computation or a large amount of memory.

!!! important "Rule of thumb"
    Never run a program that takes more than a few seconds on a login node,
    unless it mostly sleeps or mostly moves data.

The following non-exhaustive list gives a sense of what is and is not allowed
on the login nodes:

!!! do "Acceptable"
    - A Python or shell script that downloads a dataset and exits immediately
    after may be acceptable. *(Mostly moves data)*
    - A Python hyperparameter search script that uses `submitit` to launch jobs,
    and only sleeps waiting for them to end and run other jobs, is acceptable.
    *(Mostly sleeps; the actual jobs run on the compute nodes)*
    *(Takes far too much RAM to compile the CUDA kernels)*
    - Editing code with `nano`, `vim` or `emacs` is acceptable.
    *(Editors mostly sleep awaiting user keystrokes)*
    - Copying or moving files with `cp`, `mv`, etc. is acceptable.
    *(Mostly moves data)*
    - Connecting to compute nodes with `ssh` is acceptable.
    *(Mostly sleeps, forwarding keystrokes and ports to and from the node)*
    - Using `tmux` is acceptable.
    *(Mostly sleeps, managing the processes under its control)*

!!! dont "Unacceptable"
    - A Python training script.
    *(Too computationally- and memory-intensive)*
    - `pip install` of packages like `vllm` or `flash-attn` from source code. *(Takes too much RAM to compile the CUDA kernels)*.

## Connect to compute nodes

### Connect to a node running a job

SSH access to a compute node `cnode` is allowed if, and only if, a job of
yours is running on it, for example to open a second terminal. The session
ends automatically when the job ends.

First, add the public key (the one provided to IT support) to the
`~/.ssh/authorized_keys` file on the cluster, or configure an `ssh-agent`
that forwards that key when connecting to the compute node.

```bash
# ON A LOGIN NODE
mkdir -p ~/.ssh
echo "THE SSH PUBLIC KEY THAT YOU GAVE TO IT-SUPPORT" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod go-w ~   # in case you accidentally gave too many permissions for $HOME in the past.
```

Then connect to the node:

- From a login node, run `ssh cnode`.
- From a local machine, run `ssh -J mila USERNAME@cnode`. The `-J` option
  "jumps" through the login node, which is necessary because the compute
  nodes are behind a firewall.

Optionally, add the following wildcard rule to `~/.ssh/config`:

```sshconfig
Host *.server.mila.quebec !*login.server.mila.quebec
    HostName %h
    User YOUR-USERNAME
    ProxyJump mila
```

This rule allows connecting to a compute node with
`ssh <node>.server.mila.quebec`.

### Auto-allocate a CPU with `mila-cpu`

After installing [milatools](#mila-init) and running `mila init`, allocate a
CPU on a compute node and connect to it automatically with:

```bash
ssh mila-cpu
```

Multiple connections to `mila-cpu` all reuse the same job, so use it
liberally. It also works transparently with the VSCode Remote SSH feature.

Use `mila-cpu` for light work that is too heavy for a login node but does not
require many resources: editing in VSCode, building conda environments,
running tests, etc.

!!! warning
    The `mila-cpu` entry must be present in `~/.ssh/config`. Modify it at
    your own risk.

## Use a non-Bash Unix shell

Bash is the standard shell on the cluster, and the cluster is designed to
support both Bash and Zsh. Mila does not provide support for debugging shell
setups, but if something should work with Zsh and does not, contact
[Mila's IT support](https://it-support.mila.quebec).

## Next steps

<div class="grid cards" markdown>

-   [:material-language-python:{ .lg .middle } __Manage Python Dependencies with `uv`__](python_uv.md)
    { .card }

    ---
    Install uv, manage project dependencies, run reproducible Slurm jobs, and run
    standalone scripts.

-   [:material-key:{ .lg .middle } __Secure Shell (SSH)__](../../technical_reference/general_theory/ssh_on_clusters.md)
    { .card }

    ---
    How SSH keys and MFA work, and how to generate an SSH key pair.

</div>
