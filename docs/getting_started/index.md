# Get Started with the Cluster

This guide walks you through obtaining a Mila account, connecting to the cluster, and
setting up the tools you need to run jobs. Follow the steps in order.

---

## Obtain your Mila account (`@mila.quebec`) { #obtain-your-mila-account }

1. Ask your supervisor how to get invited into the Mila organization and obtain
   your `@mila.quebec` account.
2. After your supervisor submits the application, you will receive a
   confirmation email from IT support with instructions to access your account
   and connect to the cluster.

!!! tip "Still waiting for your account?"
    If you feel like this is taking longer than expected, contact [MyMila
    support](https://mila-iqia.atlassian.net/servicedesk/customer/portal/8).

## Enable your cluster access { #enable-your-cluster-access }

1. Read the [IT Onboarding
   Guide](https://sites.google.com/mila.quebec/mila-intranet/it-infrastructure/it-onboarding-training)
   and complete and submit the quiz.
2. Once you'll pass the quiz, you'll get contacted either by email or on Slack
   with the information to connect to the cluster for the first time. Take note
   of your cluster's username. It can take up to 48h for your access to the
   cluster to be effective.
3. An email will be sent for you to activate your Multi-Factor Authentication.

## Set up Multi-Factor Authentication (MFA)

Cluster access requires **two factors**: your SSH key (first factor) and a
second factor (TOTP, push notification, or email token). You must complete MFA
setup before you can connect via SSH.

### Get your registration token

You will receive an automated email with a **one-time registration token**. Use
it as soon as possible; it expires after use.

### First-time MFA setup

1. Go to **https://mfa.mila.quebec**.

    ![Login-interface](_static/screenshots/mfa-login.png)

2. **Username:** your cluster username.

3. **Password:** enter the **registration token** from the email (not your
   account password).

4. After logging in, **immediately** add at least one **TOTP** token to your
   account:

    ![Token-selector](_static/screenshots/mfa-enroll-token-totp.png)

    1. If you don't have a TOTP authenticator app yet, install one on your phone
       first:

        - privacyIDEA Authenticator
          ([:material-android:](https://play.google.com/store/search?q=privacyidea%20authenticator&c=apps)
          /
          [:material-apple:](https://apps.apple.com/iphone/search?term=privacyidea%20authenticator)).
        - Authy
          ([:material-android:](https://play.google.com/store/search?q=authy&c=apps)
          /
          [:material-apple:](https://apps.apple.com/iphone/search?term=authy)).
        - Google Authenticator
          ([:material-android:](https://play.google.com/store/search?q=google%20authenticator&c=apps)
          /
          [:material-apple:](https://apps.apple.com/iphone/search?term=google%20authenticator)).
        - Microsoft Authenticator
          ([:material-android:](https://play.google.com/store/search?q=microsoft%20authenticator&c=apps)
          /
          [:material-apple:](https://apps.apple.com/iphone/search?term=microsoft%20authenticator)).

    2. In your authenticator app, scan the QR code shown on the MFA page to add
       the token:

        ![Token-selector](_static/screenshots/mfa-enroll-token-totp-2.png)

!!! warning "Set up TOTP before you leave"
    After this first visit, the MFA web portal **only** accepts a TOTP code. If
    you leave without setting up TOTP or Push, you will be locked out and will
    need a new registration token from [IT
    support](https://it-support.mila.quebec).

---

???+ warning "I'm using :material-microsoft-windows-classic:Windows, how do I install WSL?"

    Windows users need [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/) to run the commands in this guide (`curl`, `ssh`, `uv`, etc.).

    **Steps:**

    1. Open PowerShell.
    2. Run:
       ```bash
       wsl --install Ubuntu
       ```
    3. Restart your computer when prompted.
    4. After restart, WSL will finish setup. You may be asked to create a Linux username and password.
    5. Open **Ubuntu** from the Start menu to get a Linux terminal.

    **Verify:** In the WSL terminal, run `ls` and `curl --version` to confirm you have a working shell.

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
        All commands in this guide (`uv`, `milatools`, `ssh`) should be run
        inside the WSL terminal, not in Windows PowerShell or Command Prompt.

---

## Install `uv` on your local machine

`uv` is a fast Python package manager and workflow tool that serves as a drop-in
replacement for `pip` and `virtualenv`, allowing you to quickly install project
dependencies, manage packages, and create isolated Python environments.

On your **local machine**, run:

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

**Prerequisite:** It is required to complete the [Obtain your Mila
                  account](#obtain-your-mila-account) and [Enable your cluster
                  access](#enable-your-cluster-access) sections before
                  connecting to the cluster.

### Install `milatools`

Install `milatools` locally (after [`uv` is installed on your local machine](#install-uv-on-your-local-machine)):

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

Run `mila init` with your username and password ready. This sets up your SSH config, public keys, and passwordless auth.

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

Open a terminal and run:

```bash
ssh mila
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
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

You should land on a Mila login node. If this works, your connection is set up
correctly. If not, check again the steps to [install and configure
`milatools`](#install-milatools)

## Install `uv` on the cluster

Once you can [connect via SSH to the Mila cluster](#verify-your-connection), run
the same install command as before but on a **login node**:

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

## Next steps

Once you have access to the cluster, head to the following sections to run your
first job and train your first model:

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Run Your First Job__](../getting_started/my_first_job/)
    { .card }

    ---

    Run your first job on the cluster with PyTorch using VSCode on a GPU compute
    node.

-   [:material-run-fast:{ .lg .middle } __Train Your First Model__](../getting_started/train_first_model/)
    { .card }

    ---

    Train a ResNet18 on CIFAR-10 on a single GPU using `sbatch`.

</div>