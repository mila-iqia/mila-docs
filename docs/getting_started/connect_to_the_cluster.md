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

## What this guide covers

* Understand which token to use for the portal vs. SSH cluster logins
* Complete a cluster login after MFA is active

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
    
    - [Obtain your Mila account](#obtain-your-mila-account)
    - [Enable your cluster access](#enable-your-cluster-access)
    - [Set up Multi-Factor Authentication (MFA)](#set-up-mfa)

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
TOTP code from the [authenticator app](#set-up-mfa) — *the code will not appear
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