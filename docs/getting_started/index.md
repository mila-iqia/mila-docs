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
        <div class="progress-marker" aria-current="step">1</div>
        <div class="progress-label">Enable your cluster access</div>
    </div>
    <div class="progress-step">
        <div class="progress-marker" aria-current="step">2</div>
        <div class="progress-label">Set up MFA</div>
    </div>
    <div class="progress-step">
        <div class="progress-marker">3</div>
        <div class="progress-label">Connect to the cluster</div>
    </div>
    <div class="progress-step">
        <div class="progress-marker">4</div>
        <div class="progress-label">Run your first job</div>
    </div>
    <div class="progress-step">
        <div class="progress-marker">5</div>
        <div class="progress-label">Train your first model</div>
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
