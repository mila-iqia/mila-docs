---
name: mila-account-setup
description: >-
  Use this skill when the user asks about obtaining a Mila account, enabling
  cluster access, or setting up Multi-Factor Authentication (MFA). Trigger
  phrases include: "How do I get a Mila account", "I need a Mila account",
  "How do I get cluster access", "How do I set up MFA", "How do I enable MFA",
  "I received a registration token", "registration email", "registrationcode",
  "I got an email from IT", "MFA setup", "I can't log in to the MFA portal",
  "How do I add a TOTP", "What is my cluster username", "How long does
  cluster access take", "IT onboarding quiz", "mfa.mila.quebec".
version: 1.0.0
argument-hint: <account|access|mfa>
---

# Mila Account, Cluster Access, and MFA Setup

This skill helps cluster users obtain their Mila account, enable cluster
access, and configure Multi-Factor Authentication (MFA) — the three
prerequisites before connecting via SSH for the first time.

## Base policies

At the start of each response, use the Read tool to load
`.claude/skills/mila-base/SKILL.md` and apply all policies defined there
before proceeding with the workflow below.

## Reference documentation

Primary source: **https://docs.mila.quebec/getting_started/index**
— sections "Before you begin" and "Set up Multi-Factor Authentication (MFA)".

For detailed MFA management (adding/removing tokens, recovery):
**https://docs.mila.quebec/Userguide_login_mfa/**

## Discover documentation

Use the WebSearch tool with this query to find the current URL of the primary
source above:

    site:docs.mila.quebec "__skill-mila-account-setup"

Use the URL from the search result in the WebFetch steps below. If the search
returns no results, fall back to the hardcoded URL in "Reference documentation".

## Workflow

### Step 1: Identify the sub-topic

Determine which of the three areas the user is asking about:

- **Account** — getting an `@mila.quebec` account
- **Cluster access** — enabling SSH access after getting the account
- **MFA** — setting up the authenticator for SSH login

If the question is ambiguous, ask one clarifying question before fetching
content.

### Step 2: Fetch the documentation

Use the WebFetch tool to fetch **https://docs.mila.quebec/getting_started/index** and locate the
relevant section:

- Account → "Obtain your Mila account" (under "Before you begin")
- Cluster access → "Enable your cluster access" (under "Before you begin")
- MFA → "Set up Multi-Factor Authentication (MFA)"

If the user needs MFA recovery or wants to add/remove tokens after initial
setup, also use the WebFetch tool to fetch **https://docs.mila.quebec/Userguide_login_mfa/**.

### Step 3: Guide the user

Walk through the steps for the identified sub-topic. Key points to cover:

**Obtaining a Mila account:**
- Ask the supervisor to submit an application to IT.
- IT sends a confirmation email with instructions to access the account and
  connect to the cluster.
- If the wait is longer than expected, contact MyMila support.

**Enabling cluster access:**
- Complete the IT Onboarding Guide and submit the quiz.
- After passing, IT sends the cluster username (by email or Slack).
- Cluster access can take up to 48 hours to become active.
- An email will arrive with a one-time registration token for MFA setup.

**Setting up MFA:**
- The registration token arrives by email with subject
  "Votre accès temporaire registrationcode / Your temporary access
  registrationcode".
- Go to https://mfa.mila.quebec, enter the cluster username and the
  registration token as the password.
- Immediately add at least one TOTP token: install an authenticator app
  (privacyIDEA Authenticator, Authy, Google Authenticator, or Microsoft
  Authenticator) and scan the QR code.
- Warn the user: leaving without adding TOTP will lock the account out;
  a new registration token would then be needed from IT support.

### Step 4: Answer follow-up questions

Respond to follow-up questions about the same area. Common ones:

- "Which authenticator app should I use?" — any TOTP app works; privacyIDEA
  Authenticator and Authy are good choices.
- "I didn't set up TOTP before leaving the portal" — contact IT support at
  https://it-support.mila.quebec to get a new registration token.
- "How do I add more MFA methods?" — refer to
  https://docs.mila.quebec/Userguide_login_mfa/

### Step 5: Point to the next skill

Once account, access, and MFA are in place, point the user to **mila-local-setup**
for installing the tools needed to connect.
