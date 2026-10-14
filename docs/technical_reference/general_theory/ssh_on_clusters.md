---
title: Secure Shell (SSH)
description: In-depth knowledge about SSH
---

<!-- START -->
# Secure Shell (SSH)

## SSH authentication on the cluster

When connecting to the cluster via SSH, authentication proceeds in
three steps:

```mermaid
sequenceDiagram
    participant C as Local machine
    participant S as Cluster (login node)
    participant M as MFA service

    C->>S:      Connection request (public key identity)
    S-->>C:     Key challenge
    C->>S:      Signed challenge response
        S->>M:  Request MFA verification
    M-->>C:     Prompt (TOTP/email code or Push notification)
    alt TOTP or email
        C->>S: 6-digit code
    else Push notification
        C->>S: Approve on phone
    end
    S-->>C:     Access granted
```

1. **Key exchange** — the local machine proves possession of the private key
   matching the public key stored on the cluster.
2. **MFA challenge** — once the key is accepted, the cluster prompts for the
   second factor.
3. **Validation** — for Push, tap "Approve" on the smartphone; for TOTP or
   email, type the code into the terminal prompt.


## Key exchange

SSH key authentication is a technique using pairs of closely-linked keys: A
private key, and a corresponding public key. The public key should be
distributed to everyone, while the private key is known to only one person. The
public key can be used by anyone to challenge a person to prove their identity.
If they have the corresponding private key, that person can perform an
electronic signature that everyone can validate but that no one else could have
done themselves. The challenge is thus answered by demonstrating possession of
the private key (and therefore their identity), without ever revealing the
private key itself.

*Mila asks you to generate a pair of SSH keys, to provide Mila only with your
public key,* which has no confidentiality implications, and to keep the private
key for yourself. *The private key must remain secret and solely known to you,*
because anyone who possesses it is capable of impersonating you by performing
your electronic signature.

During the IT Onboarding Training, you will be asked to submit that SSH
public key.
Mila allows a maximum of 4 different public keys per user.

- If you do not know what SSH keys are, or are not familiar with them, you can
  read the informative material below, then proceed to
  generate them.
- If you do not already have SSH keys, or are not sure if you have them, skip
  to the instructions on how to generate them [here](#checking-if-you-already-have-ssh-private-keys).

### SSH Private Keys

A private SSH key commonly takes the form of an obscure text file. It encodes
the digital secret of how to make an electronic signature — specifically, yours.
The content of a private SSH key might resemble

```
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
NhAAAAAwEAAQAAAYEAl5dD/UU2CvauaVS2/4/iWoUyO1Hey+m8KojCFMvIywL6PPdYRqVa
FOidmOw/E9V2HVzHz/z/2Dj6TO5xNX1qJFk7A/ACGGc1+KguIDQWdjR6AZb5Tat+aAMYro
…
aSeJOS59knbQJeBwPm0g5G+iFz6R17446dXk5jn3/29AutF5MPnKwqE0mjywxCLYxVX3He
YSOCZfE80P/z4sImW82BYxAzKtI8kKagLmHS4gXJEmE13Dfyq0xcB3q5OMuQ2fZwvukTx3
xdWgyqFrMyC4wHAAAAAAEC
-----END OPENSSH PRIVATE KEY-----
```

In the real world, a handwritten signature is useless for authenticating you if
it can be easily reproduced by others. In the virtual world, the same is true.
Anyone who has your private key is capable of reproducing your electronic signature.
It is therefore essential that only one person — you — holds this private key.
The secrecy of the private key is the guarantor of your online identity.

Mila will ***never*** ask you for your private SSH key, and any pretense of
request for a private key constitutes an attempt at phishing and identity theft.
Keep your private keys safe and do not share them with anyone. Do not put them
in the cloud, your emails, Slack messages, or Git repos. Protect them with a
passphrase.

### SSH Public Keys

A public SSH key is a simple line of text, albeit sometimes very long, commonly
found in a file with the `.pub` extension. It encodes the digital knowledge
required to recognize and validate your electronic signature, without however
making it possible to reproduce it elsewhere. Here are three examples of public
SSH keys:

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDMYpSndal/…mPL+NXs=
ssh-ed25519 AAAA…d/ca2h  user@server
ecdsa-sha2-nistp256 AAAA…hWQcQg8=  mylaptop
```

You are requested to submit just such a public SSH key to Mila, which will
allow Mila to recognize you when you connect to the Mila cluster, but without
revealing the secret of how to perform your signature.

### Checking If You Already Have SSH (Private) Keys

Usually, a private SSH key is found in the hidden directory `~/.ssh/` and is
named `id_rsa`, `id_ed25519`, or `id_ecdsa`. Its corresponding public SSH
key is usually in the same directory and shares the same filename, except with
a `.pub` suffix.

### Generating an SSH Private Key

If no private SSH key already exists, you can create one with the `ssh-keygen` utility:



| RSA                                                                                                                                                     | Ed25519                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| *Integer factorization*                                                                                                                                 | *Elliptic curve*                                                                         |
| <ul><li>Classic</li><li>Ultra-compatible, standardized, the reference</li><li>Large key size, but configurable</li><li>Slow or even very slow</li></ul> | <ul><li>New</li><li>Less compatible</li><li>Fixed, small key size</li><li>Fast</li></ul> |  | * New |
| `$ ssh-keygen -t rsa -b 3072`                                                                                                                           | `$ ssh-keygen -t ed25519`                                                                |

!!! tip
    The pass-**phrase** protects the SSH private key **on-disk**. The
    passphrase is **not** the same thing as the pass-**word** used to *log into
    your personal computer account*. However, choosing them to be equal may
    allow for automatic unlocking of encrypted SSH private keys at login, in
    combination with software such as ``pam_ssh(8)`` (Linux) or Keychain
    (Mac OS X/macOS). This makes the good practice of using encrypted keys
    convenient as well.

### Generating an SSH Public Key from a Private Key

If a private SSH key exists, but not its corresponding SSH public key, it can
be recalculated with the `ssh-keygen` utility as well:

| RSA                                                    | Ed25519                                |
| ------------------------------------------------------ | -------------------------------------- |
| **SSH public key:**                                    | **SSH public key:**                    |
| <ul><li>380 bytes @ 2048 bits (not rec.)    </li></ul> | ~82 bytes                              |
| <ul><li>550 bytes @ 3072 bits (recommended) </li></ul> |                                        |
| <ul><li>725 bytes @ 4096 bits (slower)      </li></ul> |                                        |
| <ul><li>1400 bytes @ 8192 bits (much slower)</li></ul> |                                        |
| `$ ssh-keygen -y -f ~/.ssh/id_rsa`                     | `$ ssh-keygen -y -f ~/.ssh/id_ed25519` |

It is this SSH public key that you should submit in the IT Onboarding Training form.


## MFA challenge and validation

Multi-Factor Authentication (MFA) adds a security layer beyond SSH keys.
A MFA token has to be set to access the cluster (see [Getting started instructions](../../getting_started/mfa.md)).
