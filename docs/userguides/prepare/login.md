---
title: Logging in to the cluster
description: >-
  Configure SSH, log in to the Mila login nodes and connect to compute nodes.
---

# Logging in to the cluster

This guide describes the advanced ways to connect to the Mila cluster:
configuring SSH automatically or manually, logging in to a login node,
completing the Multi-Factor Authentication (MFA) prompt and connecting to
compute nodes. For a quick and simple way to connect, follow the [Getting started](../../getting_started/index.md) section instead.

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

- [`mila init`](#configure-ssh-with-mila-init) automatically adds a simple SSH configuration to
- `~/.ssh/config`. Advanced users can alternatively edit `~/.ssh/config`
[manually](#manual-ssh-configuration).

### Configure SSH with `mila init`

Mila publishes the [milatools](https://www.github.com/mila-iqia/milatools)
package to help set up a productive environment. Its `mila init` command
performs some of the steps below automatically. It requires Python 3.9 or
later:

```bash
pip install milatools
mila init
```

!!! tip
    If you encounter problems with milatools, try updating it:

    - run `uv tool upgrade milatools` or `pip install -U milatools`
    - then re-run `mila init` to apply new features and bug fixes.

### Manual SSH configuration

The Mila **login** nodes require:

- `Hostname`: `login.server.mila.quebec`
- `Port`: `2222`
- `User`: *your Mila account username*
- `PreferredAuthentications`: `publickey,keyboard-interactive`

You can use the following template as a starting point for
the entries in `~/.ssh/config`.

!!! warning "Edit the template"
    Note that you:

    - have to change the usernames (`CHANGE_TO_YOUR_MILA_USERNAME` and/or `CHANGE_TO_YOUR_DRAC_USERNAME`) to your cluster username
    - may have to change the SSH key path (`~/.ssh/id_rsa_mila`) or generate such a key (see [Generating an SSH Private Key](../../technical_reference/general_theory/ssh_on_clusters.md#generating-an-ssh-private-key))

=== "Mila cluster configuration"
    ```sshconfig
    Host mila
        HostName login.server.mila.quebec
        PreferredAuthentications publickey,keyboard-interactive
        Port 2222
        ServerAliveInterval 120
        ServerAliveCountMax 5
        ControlMaster auto
        ControlPath ~/.cache/ssh/%r@%h:%p
        ControlPersist yes
        User CHANGE_TO_YOUR_MILA_USERNAME
        IdentityFile ~/.ssh/id_rsa_mila

    Host mila-cpu
        Port 2222
        ForwardAgent yes
        StrictHostKeyChecking no
        LogLevel ERROR
        UserKnownHostsFile /dev/null
        RequestTTY force
        ConnectTimeout 600
        ServerAliveInterval 120
        ProxyCommand ssh mila "/cvmfs/config.mila.quebec/scripts/milatools/slurm-proxy.sh mila-cpu --mem=8G"
        RemoteCommand /cvmfs/config.mila.quebec/scripts/milatools/entrypoint.sh mila-cpu
        ControlMaster auto
        ControlPath ~/.cache/ssh/%r@%h:%p
        ControlPersist yes
        User CHANGE_TO_YOUR_MILA_USERNAME
        IdentityFile ~/.ssh/id_rsa_mila

    Host *.server.mila.quebec !*login.server.mila.quebec
        ProxyJump mila
        ControlMaster auto
        ControlPath ~/.cache/ssh/%r@%h:%p
        ControlPersist yes
        User CHANGE_TO_YOUR_MILA_USERNAME
        IdentityFile ~/.ssh/id_rsa_mila

    Host cn-????
        ProxyJump mila
        ControlMaster auto
        ControlPath ~/.cache/ssh/%r@%h:%p
        ControlPersist yes
        User CHANGE_TO_YOUR_MILA_USERNAME
        IdentityFile ~/.ssh/id_rsa_mila
    ```

=== "DRAC/PAICE clusters configuration"
    ```sshconfig
    Host narval rorqual fir nibi trillium trillium-gpu tamia killarney vulcan
        HostName %h.alliancecan.ca
        ControlMaster auto
        ControlPath ~/.cache/ssh/%r@%h:%p
        ControlPersist yes
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host nc????? ng?????
        ProxyJump narval
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host rc????? rg????? rl?????
        ProxyJump rorqual
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host fc????? fb?????
        ProxyJump fir
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host c? c?? c??? g? g?? l? l?? m? m?? u?
        ProxyJump nibi
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host tg????? tc?????
        ProxyJump tamia
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host kn???
        ProxyJump killarney
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host rack??-??
        ProxyJump vulcan
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host !trillium tri????
        ProxyJump trillium
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac

    Host !trillium trig????
        ProxyJump trillium-gpu
        User CHANGE_TO_YOUR_DRAC_USERNAME
        IdentityFile ~/.ssh/id_rsa_drac
    ```

Here are some bearings if you are not familiar with configuration files:

| Option | What it does |
|---|---|
| `Host mila` | Creates the `mila` alias, so you can type `ssh mila`, `scp … mila:…`, `rsync … mila:…`. This is later use in the file with `ProxyJump mila` |
| `HostName login.server.mila.quebec` | The real server behind the alias. It is a load balancer that picks the least busy of the 4 login nodes. |
| `Port 2222` | Mila's SSH server listens on 2222, not the default port 22. |
| `PreferredAuthentications publickey,keyboard-interactive` | Tries the SSH key first, then uses an interactive prompt. That prompt is where the MFA code or Push confirmation is asked for. |
| `IdentityFile ~/.ssh/id_rsa_mila` | The private key to use. |

More advanced parameters are:

| Option | What it does |
|---|---|
| `ControlMaster auto` | This is connection multiplexing: the first connection becomes a "master" and later `ssh`/`scp`/`rsync` commands reuse it. |
| `ControlPersist yes` | Keeps the master connection open in the background after you close the first session. It is used along `ControlMaster auto` to only have to enter MFA once. See tip in [Complete the MFA prompt](#complete-the-mfa-prompt) |
| `ControlPath ~/.cache/ssh/%r@%h:%p` | Where the master connection's socket file goes, one per user, host and port. |

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
    Run commands on the login node with `ssh` directly, for example
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

Login nodes are merely *entry points* to the cluster. They give access to the
compute nodes and to the filesystem, but they are not meant to run anything
heavy. Do **not** run compute-heavy programs on these nodes: doing so could
bring them down and impede cluster access for everyone.

!!! important "Rule of thumb"
    Never run a program that takes more than a few seconds on a login node,
    unless it mostly sleeps or mostly moves data.

The following non-exhaustive list gives a sense of what is and is not allowed
on the login nodes:

!!! do "Acceptable"
    - A Python or shell script that downloads a dataset and exits immediately
    after. *(Mostly moves data)*
    - A Python hyperparameter search script that uses `submitit` to launch jobs,
    and only sleeps waiting for them to end and run other jobs.
    *(Mostly sleeps; the actual jobs run on the compute nodes)*
    - Editing code with `nano`, `vim` or `emacs`.
    *(Editors mostly sleep awaiting user keystrokes)*
    - Copying or moving files with `cp`, `mv`, etc.
    *(Mostly moves data)*
    - Connecting to compute nodes with `ssh`.
    *(Mostly sleeps, forwarding keystrokes and ports to and from the node)*
    - Using `tmux`.
    *(Mostly sleeps, managing the processes under its control)*

!!! dont "Unacceptable"
    - A Python training script.
    *(Too computationally- and memory-intensive)*
    - `pip install` of packages like `vllm` or `flash-attn` from source code. *(Takes too much RAM to compile the CUDA kernels)*.
    - VSCode remote SSH instances *(Even though they are typically not very computationally expensive, when many people do it, they add up. See [Visual Studio Code](../../toolbox/development/VSCode.md) for specific instructions.)*

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
- From a local machine, run `ssh -J mila USERNAME@cnode`.

!!! info
    The `-J` option "jumps" through the login node, which is necessary because the compute
    nodes are behind a firewall.

    However, if your SSH configuration file (`~/.ssh/config`) contains `ProxyJump mila`,
    `-J mila` is done through it and does not have to be added (see example [here](#manual-ssh-configuration)).

This rule allows connecting to a compute node with
`ssh <node>.server.mila.quebec`.

### Auto-allocate a CPU with `mila-cpu`

After installing [milatools](#configure-ssh-with-mila-init) and running `mila init`, allocate a
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
