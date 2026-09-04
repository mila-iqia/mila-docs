---
title: Multi-Factor Authentication (MFA)
description: Configure MFA and use it to access the Mila cluster securely.
---

<!-- START -->
# Set Up Multi-Factor Authentication

Multi-Factor Authentication (MFA) adds a security layer beyond SSH keys.
After setup, every cluster login requires two distinct factors: an SSH
public key (first factor) and a dynamic verification code (second
factor). This guide covers how to register for MFA, choose an
authentication method, and complete a cluster login.

<nav class="progress-track" aria-label="Getting started progression">
    <div class="progress-step is-done">
        <div class="progress-marker">✓</div>
        <div class="progress-label">Enable your cluster access</div>
    </div>
    <div class="progress-step is-current">
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


## What this guide covers

* Choose a second-factor authentication method
* Register on the MFA web portal using an email registration token

---

## Set up Multi-Factor Authentication (MFA) { #set-up-mfa }

Cluster access requires **two factors**: an SSH key (first factor) and a second
factor (TOTP, push notification, or email token). The MFA setup **must** be
completed before connecting via SSH.

### Get your registration token

Look for an email with the subject *Votre accès temporaire registrationcode /
Your temporary access registrationcode*; it contains a **one-time registration
token** that expires after use.

### First-time MFA setup

!!! warning "Set up TOTP before leaving"
    After the first visit, the MFA web portal will **only** accepts a TOTP code.
    Leaving without setting up TOTP locks out the account, and a new
    registration token will be needed from [IT
    support](https://it-support.mila.quebec).

1. Go to **https://mfa.mila.quebec**.

    ![Login-interface](../_static/screenshots/mfa-login.png)

2. **Username:** your cluster username (**not** your `@mila.quebec` email
   address).

3. **Password:** enter the **registration token** from the email (**not** your
   account password).

4. After logging in, **immediately** add at least one **TOTP** token to your
   account:

    ![Token-selector](../_static/screenshots/mfa-enroll-token-totp.png)

    1. Install a TOTP authenticator app:

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

    2. In the authenticator app, scan the QR code shown on the MFA page to add
       the token:

        ![Token-selector](../_static/screenshots/mfa-enroll-token-totp-2.png)

## Troubleshooting

**TOTP codes rejected**
:   TOTP codes are time-sensitive. Set the smartphone clock to
    automatic time synchronization to keep codes valid.

**Lost phone or device**
:   Contact [IT Support](https://it-support.mila.quebec) immediately
    to reset MFA tokens.

---

## Next step

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Log in to the cluster__](userguides/login.md)
    { .card }

    ---
    Connect to the Mila cluster via SSH with MFA configured.

&nbsp;

</div>
