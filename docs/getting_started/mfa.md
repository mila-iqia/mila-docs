---
title: Multi-Factor Authentication (MFA)
description: Configure MFA to access the Mila cluster securely.
---

<!-- START -->
# Set Up Multi-Factor Authentication

<nav class="progress-track" aria-label="Getting started progression">
    <div class="progress-step is-done">
        <div class="progress-marker"><a href="../cluster_access">✓</a></div>
        <div class="progress-label"><a href="../cluster_access">Enable your cluster access</a></div>
    </div>
    <div class="progress-step is-current">
        <div class="progress-marker" aria-current="step"><a href="../mfa">2</a></div>
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

This guide covers how to register for MFA and choose an
authentication method to be able to complete a cluster login.


## What this guide covers

* Choose a second-factor authentication method
* Register on the MFA web portal using an email registration token

---

## Set up Multi-Factor Authentication (MFA) { #set-up-mfa }

Multi-Factor Authentication (MFA) adds a security layer beyond SSH keys.

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

5. You can then add other registration tokens (PUSH (recommended for recurrent cluster access), TOTP or email).
    See next section for more details.

    !!! tip "Add PUSH to lighten cluster connection"
        The PUSH option is used to confirm access with a phone without having
        to enter a code on your computer. It simplifies the procedure and could
        save time along the day.


## Subsequent logins to the portal

After the first session, the portal accepts **TOTP tokens only**.
Email tokens can no longer be used to access the portal — they remain
valid only for SSH cluster logins.

**Which token to use**

| Access type    | First login                | Every subsequent login |
| -------------- | -------------------------- | ---------------------- |
| MFA web portal | Email registration token   | TOTP token only        |
| Cluster SSH    | N/A                        | TOTP, Push, or email   |

??? info "Authentication methods overview"

    **PrivacyIDEA Push notification**
    :   Approve a login request via a push notification on a smartphone.
        Requires the **privacyIDEA Authenticator** app (iOS or Android).

    **TOTP (Time-based One-Time Password)**
    :   Enter a 6-digit rolling code from an authenticator app. Compatible
        with **privacyIDEA**, Google Authenticator, Microsoft Authenticator,
        or any app supporting the RFC 6238 standard.

    **Email token**
    :   Receive a one-time verification code at the registered
        **@mila.quebec** email address.

    **Hardware token (coming soon)**
    :   YubiKey support (FIDO2/WebAuthn) is planned for a future update.


## Troubleshooting

**TOTP codes rejected**
:   TOTP codes are time-sensitive. Set the smartphone clock to
    automatic time synchronization to keep codes valid.

**Lost phone or device**
:   Contact [IT Support](https://it-support.mila.quebec) immediately
    to reset MFA tokens.


---

## Key concepts

**MFA**
:   Multi-Factor Authentication (MFA) adds a security layer beyond SSH keys.
    After setup, every cluster login requires two distinct factors: an SSH
    public key (first factor) and a dynamic verification code (second
    factor).

---

## Next step

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Connect to the cluster__](connect_to_the_cluster.md)
    { .card }

    ---
    Connect to the Mila cluster via SSH with MFA configured.

&nbsp;

</div>
