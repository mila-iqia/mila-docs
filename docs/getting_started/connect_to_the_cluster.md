---
title: Connect to the Cluster
description: Install tools to connect to the cluster for the first time.
---

<!-- START -->
<nav class="progress-track" aria-label="Getting started progression">
    <div class="progress-step is-done">
        <div class="progress-marker"><a href="../cluster_access">✓</a></div>
        <div class="progress-label"><a href="../cluster_access">Enable your cluster access</a></div>
    </div>
    <div class="progress-step is-done">
        <div class="progress-marker"><a href="../mfa">✓</a></div>
        <div class="progress-label"><a href="../mfa">Set up MFA</a></div>
    </div>
    <div class="progress-step is-current">
        <div class="progress-marker" aria-current="step"><a href="../connect_to_the_cluster">3</a></div>
        <div class="progress-label"><a href="../connect_to_the_cluster">Connect to the cluster</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="../my_first_job">4</a></div>
        <div class="progress-label"><a href="../my_first_job">Run your first job</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="../train_first_model">5</a></div>
        <div class="progress-label"><a href="../train_first_model">Train your first model</a></div>
    </div>
</nav>

This guide helps with the installation of `uv` and `milatools`, in order to connect to the cluster.


## What this guide covers

* Understand which token to use for the portal vs. SSH cluster logins
* Complete a cluster login once MFA is active


[](){ #install-wsl }

???+ warning ":material-microsoft-windows-classic: Windows users: install WSL first"

    Windows users need [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/) to run the commands in this guide (`curl`, `ssh`, `uv`, etc.).

    **Steps:**

    1. Open PowerShell.
    2. Run:
       ```bash
       wsl --install Ubuntu
       ```
    3. Restart the computer when prompted.
    4. After restart, WSL will finish setup. A prompt may appear to create a
       Linux username and password.
    5. Open **Ubuntu** from the Start menu to get a Linux terminal.

    **Verify:** In the WSL terminal, run `ls` and `curl --version` to confirm
    the shell is functional.

    === "`ls`"
        ```bash
        ls
        ```
        <div class="result" style="border:None; padding:0" markdown>
        ``` linenums="0"
        bin      CODE     scratch
        ```
        </div>

    === "`curl --version`"
        ```bash
        curl --version
        ```
        <div class="result" style="border:None; padding:0" markdown>
        ``` linenums="0"
        curl 8.4.0 (x86_64-pc-linux-gnu) libcurl/8.4.0 OpenSSL/3.0.9 zlib/1.2.13 brotli/1.0.9 zstd/1.5.5 c-ares/1.19.1 nghttp2/1.51.0
        Release-Date: 2023-10-11
        Protocols: dict file ftp ftps http https imap imaps mqtt pop3 pop3s rtsp smtp smtps tftp
        Features: alt-svc AsynchDNS brotli HSTS HTTP2 HTTPS-proxy IPv6 Largefile libz NTLM SSL threadsafe TLS-SRP UnixSockets zstd
        ```
        </div>

    ???+ info "References"
        1. [Ubuntu WSL install guide](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/)
        2. [Microsoft WSL install guide](https://learn.microsoft.com/en-us/windows/wsl/install)

    !!! note
        Run all commands in this guide (`uv`, `milatools`, `ssh`) inside the
        WSL terminal, not in Windows PowerShell or Command Prompt.


## Install `uv` on a local machine

`uv` is a fast Python package manager and workflow tool, that serves as a
drop-in replacement for `pip` and `virtualenv`, for quickly installing project
dependencies, managing packages, and creating isolated Python environments.

On a **personal computer**, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
downloading uv 0.10.10 x86_64-unknown-linux-gnu
no checksums to verify
installing to /home/username/.local/bin
  uv
  uvx
everything's installed!
```
</div>

???+ info "References"
    1. [uv documentation](https://docs.astral.sh/uv/)

## Connect to the cluster

!!! success "Cluster access"
    Before proceeding, complete:
    
    - [Obtain your Mila account](cluster_access.md)
    - [Enable your cluster access](cluster_access.md#enable-your-cluster-access)
    - [Set up Multi-Factor Authentication (MFA)](mfa.md)

### Install `milatools`

`milatools` is a command-line tool that simplifies connecting to the Mila
cluster. It configures SSH automatically and provides `mila code` to open VSCode
directly on a compute node.

Install a **personal computer** (after [installing `uv`](#install-uv-on-a-local-machine)):

```bash
uv tool install --upgrade milatools
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Resolved 23 packages in 510ms
Prepared 23 packages in 206ms
Installed 23 packages in 43ms
 + bcrypt==5.0.0
 + blessed==1.33.0
 [...]
 + wcwidth==0.6.0
 + wrapt==2.1.2
Installed 1 executable: mila
```
</div>

See the [milatools README](https://github.com/mila-iqia/milatools) for more details.

### Configure `milatools`

Run `mila init` with your cluster username ready. This sets up the SSH config,
public keys, and passwordless auth.

```bash
mila init           
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Checking ssh config
Created the ssh directory at /Users/username/.ssh
Created /Users/username/.ssh/config
Do you have an account on the Mila cluster? [y/n] (y): y
What's your username on the Mila cluster?
: MILA_USERNAME
The following modifications will be made to /Users/username/.ssh/config:
[...]
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
                                                     MILA SETUP                                                      
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Checking connection to the mila login nodes... 
✅ Able to `ssh mila`
❌ Local /Users/username/.ssh/id_ed25519_mila.pub is not in ~/.ssh/authorized_keys on the mila cluster, or file 
permissions are incorrect. Attempting to fix this now.
Checking connection to compute nodes on the mila cluster. This is required for `mila code` to work properly.
[18:16:21] (mila) $ mkdir -p ~/.ssh                                                                  remote_v2.py:115
[18:16:22] (mila) $ echo 'ssh-ed25519                                                                remote_v2.py:115
           XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX From home to Mila'                   
           >> ~/.ssh/authorized_keys                                                                                 
           (mila) $ chmod 600 ~/.ssh/authorized_keys                                                 remote_v2.py:115
[18:16:23] (mila) $ chmod 700 ~/.ssh                                                                 remote_v2.py:115
           (mila) $ chmod go-w ~                                                                     remote_v2.py:115
✅ Your public key is now present in ~/.ssh/authorized_keys on the mila cluster, and file permissions are correct.
✅ Local /Users/username/.ssh/id_ed25519_mila.pub is in ~/.ssh/authorized_keys on the mila cluster and file 
permissions are correct. You should now be able to connect to compute nodes with SSH.
```
</div>

## Verify your connection

Open a terminal and run `ssh mila`. When prompted for an OTP, enter the 6-digit
TOTP code from the [authenticator app](mfa.md) — *the code will not appear
on screen as it is typed*:

```bash
ssh mila
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
(username@login.server.mila.quebec) please enter otp:
================================================================================


                .:.
        .*#*: :#%%%+...-*
        :#%#: -%%%%*  :. -
   .=+*=:   .:..---  -.   -          ..             ..   ..   ..
  :%%%%%%= *%%%*...==......-=       =%%+          :%%%  *%%= .%%=
  :%%%%%%+ #%%%#   ::.     -::      =%%%=        .#%%%  .--  .%%=
   :+##*-   :-:   :  ::  .-  .:     =%%%%:       #%%%%   ::  .%%=    .:---:
   :=-  =**= .*%%#=   .: :     :    =%%+%#.     *%**%%  -%%: .%%=  :#%#++*%%+
   %%%-:%%%%:=%%%%%....-*......:*   =%%.+%#    =%# +%%  -%%: .%%=  .-:    .%%-
    :-. :==:  -+*+:   -. -    ::    =%%. *%*  -%#. +%%  -%%: .%%=  .=*#####%%=
   +%%%%*. .=+=.  -  -    -. -.     =%%. .#%+.%%-  +%%  -%%: .%%= .#%+.   .%%=
  -%%%%%%* %%%%%...==......-+       =%%.  :%%#%=   +%%  -%%: .%%= .%%+.  :*%%=
   *%%%%#: =#%#=  .::.     :.       -**.   -**+    =**  :**.  **-  :+#%%#*-**=
    .:-: -+=   =**+. ::  .-
        +%%%+ =%%%%#  .:.:
         -=-  .+##*:...-+

                * Documentation:    https://docs.mila.quebec
                * Monitoring:       https://dashboard.server.mila.quebec
                * Support:          http://it-support.mila.quebec/
                                    or email it-support@mila.quebec

================================================================================
====================== Cluster Login-node: Login-2 =======================
================================================================================

 System information as of Mon Mar 16 06:30:05 PM EDT 2026

  System load:  0.39               Processes:              1415
  Usage of /:   40.5% of 38.09GB   Users logged in:        78
  Memory usage: 70%                IPv4 address for ens18: 172.16.2.152
  Swap usage:   0%


==================== NEWS ======================================================
================================================================================

Last login: Fri Feb 27 09:29:48 2026 from 74.58.126.98
```
</div>

After entering the OTP, the session opens on a **login node** — a shared entry
point to the cluster. Login nodes are for submitting jobs and managing files,
not for running computations directly.

??? question "Not prompted to enter an OTP?"

    Review the steps to [install and configure `milatools`](#install-milatools).

??? question "The Login node banner does not appear after entering the OTP?"

    Review the steps to [set up Multi-Factor Authentication](#set-up-mfa).

## Install `uv` on the cluster

Once [connected via SSH to the Mila cluster](#verify-your-connection), run the
same `uv` install command as before but on a **login node**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
downloading uv 0.10.10 x86_64-unknown-linux-gnu
no checksums to verify
installing to /home/username/.local/bin
  uv
  uvx
everything's installed!
```
</div>


---

## Key concepts

`uv`
:   Fast Python package manager and virtual environment tool. Used on both
    a local machine and the cluster.

`milatools`
:   CLI tool (`mila`) for setting up SSH config and opening VSCode on
    compute nodes.

`SSH`
:   The Secure Shell Protocol (SSH Protocol) is a cryptographic network
    protocol for operating network services securely over an unsecured
    network. Its most notable applications are remote login and command-line
    execution.



---

## Next step

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Run your first job__](my_first_job.md)
    { .card }

    ---
    Run your first job on the Mila cluster.

&nbsp;

</div>