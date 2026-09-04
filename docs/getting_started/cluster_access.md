<nav class="progress-track" aria-label="Getting started progression">
    <div class="progress-step is-current">
        <div class="progress-marker" aria-current="step"><a href="../cluster_access">1</a></div>
        <div class="progress-label"><a href="../cluster_access">Enable your cluster access</a><</div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="../mfa">2</a></div>
        <div class="progress-label"><a href="../mfa">Set up MFA</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="../connect_to_the_cluster">3</a></div>
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


[](){ #obtain-your-mila-account }

???+ success "Obtain your Mila account (`@mila.quebec`)"

    1. Ask your supervisor how to get invited into the Mila organization and
       obtain your `@mila.quebec` account.
    2. After your supervisor submits the application, a confirmation email will
       arrive from IT support with instructions to access the account and
       connect to the cluster.
    
    !!! tip "Still waiting for your account?"
        If this takes longer than expected, contact [MyMila
        support](https://mila-iqia.atlassian.net/servicedesk/customer/portal/8).

[](){ #enable-your-cluster-access }

???+ success "Enable your cluster access"

    1. Read the [IT Onboarding
       Guide](https://sites.google.com/mila.quebec/mila-intranet/it-infrastructure/it-onboarding-training)
       and complete and submit the quiz.
    2. After passing the quiz, IT will send the connection details by email or
       on Slack, including the cluster username. Cluster access can take up to
       48 hours to become effective.
    3. IT will send an email to activate Multi-Factor Authentication.

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
