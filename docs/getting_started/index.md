---
title: Get Started with the Cluster
description: >-
  Obtain a Mila account, enable cluster access and MFA, install `uv` and
  `milatools`, configure SSH access and connect to the cluster for the first
  time.
  #__skill-mila-account-setup
  #__skill-mila-connect-cluster
  #__skill-mila-local-setup
skills:
  - __skill-mila-account-setup
  - __skill-mila-connect-cluster
  - __skill-mila-local-setup
---


# Get Started with the Cluster

This section helps you in your first steps on the cluster.



---

## Overview { #overview }

<nav class="progress-track" aria-label="Getting started progression">
    <div class="progress-step">
        <div class="progress-marker" aria-current="step"><a href="cluster_access">1</a></div>
        <div class="progress-label"><a href="cluster_access">Enable your cluster access</a></div>
        <!--<div class="progress-marker" aria-current="step"><button class="tabbed-button" tabIndex="1" aria-hidden="true">1</button></div>
        <div class="progress-label"><button class="tabbed-button" tabIndex="1" aria-hidden="true">Enable your cluster access</button></div>-->
    </div>
    <div class="progress-step">
        <div class="progress-marker" aria-current="step"><a href="mfa">2</a></div>
        <div class="progress-label"><a href="mfa">Set up MFA</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="connect_to_the_cluster">3</a></div>
        <div class="progress-label"><a href="connect_to_the_cluster">Connect to the cluster</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="my_first_job">4</a></div>
        <div class="progress-label"><a href="my_first_job">Run your first job</a></div>
    </div>
    <div class="progress-step">
        <div class="progress-marker"><a href="train_first_model">5</a></div>
        <div class="progress-label"><a href="train_first_model">Train your first model</a></div>
    </div>
</nav>

### Enable your cluster access
* [ ] Obtain your Mila account
* [ ] [Read the IT onboarding guide](https://sites.google.com/mila.quebec/mila-intranet/it-infrastructure/it-onboarding-training)
* [ ] [Submit the quiz](https://docs.google.com/forms/d/e/1FAIpQLSfVd2CGlynKQHQGxhmv6XWCt-eIm9e-Jo54xrdhE06rynsL5A/viewform)
* [ ] [Accept compute cluster terms and conditions](https://docs.google.com/forms/d/e/1FAIpQLSd_AJoVV99wLEeSP-YTmI4StZ3hI8BygaebBE8m4A8fZKB1AA/viewform)

### Set up MFA
* [ ] Get your temporary registration code
* [ ] Use it to log in [https://mfa.mila.quebec/](https://mfa.mila.quebec/)
* [ ] Install a TOTP authenticator app
* [ ] Add at least one TOTP token
* [ ] Add other token (TOTP, Push, email validation) if you wish

### Connect to the cluster
* [ ] Install `uv`
* [ ] Install `milatools`
* [ ] Run `mila init` to configure `milatools`
* [ ] `ssh` on the cluster to check your access

### Run your first job
* [ ] Install `uv` on the cluster
* [ ] Create a project directory on the cluster
* [ ] Start VSCode on a **<u>compute</u>** node
* [ ] Create the project files
* [ ] Run the script in the VSCode terminal

### Train your first model
* [ ] Create a project directory on the cluster
* [ ] Start VSCode on a **<u>compute</u>** node
* [ ] Create the project files
* [ ] Submit the job through Slurm
* [ ] Monitor the job

---

## Next steps

First of all, you need to have a Mila account and enable your cluster access:

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Enable your cluster access__](cluster_access.md)
    { .card }

    ---
    Get a Mila account and enable your cluster access.

</div>
