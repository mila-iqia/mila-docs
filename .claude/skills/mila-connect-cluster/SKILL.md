---
name: mila-connect-cluster
description: >-
  Use this skill when the user asks about connecting to the Mila cluster via
  SSH, verifying the connection, entering an OTP, or troubleshooting
  connection failures. Trigger phrases include: "How do I connect to the
  cluster", "How do I SSH into the cluster", "ssh mila",
  "login.server.mila.quebec", "How do I verify my connection",
  "I can't connect", "connection refused", "permission denied",
  "host key verification failed", "OTP", "one-time password",
  "The login node banner does not appear", "Not prompted for OTP",
  "How do I install uv on the cluster", "I'm connected, what's next".
version: 1.0.0
argument-hint: <connect|troubleshoot>
---

# Connect to the Mila Cluster via SSH

This skill guides users through connecting to the Mila cluster for the first
time, entering the OTP from their authenticator app, and troubleshooting
common connection failures.

## Base policies

At the start of each response, use the Read tool to load
`.claude/skills/mila-base/SKILL.md` and apply all policies defined there
before proceeding with the workflow below.

## Reference documentation

Primary source: **https://docs.mila.quebec/getting_started/index**
— sections "Verify your connection" and "Install uv on the cluster".

## Discover documentation

Use the WebSearch tool with this query to find the current URL of the primary
source above:

    site:docs.mila.quebec "__skill-mila-connect-cluster"

Use the URL from the search result in the WebFetch steps below. If the search
returns no results, fall back to the hardcoded URL in "Reference documentation".

## Workflow

### Step 1: Check prerequisites

Before connecting, confirm the user has:
- A Mila account and cluster username (see **mila-account-setup**)
- MFA set up with a TOTP authenticator app (see **mila-account-setup**)
- `milatools` installed and `mila init` completed (see **mila-local-setup**)

If any prerequisite is missing, point to the appropriate skill first.

### Step 2: Fetch the documentation

Use the WebFetch tool to fetch **https://docs.mila.quebec/getting_started/index** and locate the
"Verify your connection" section.

### Step 3: Guide through the SSH connection

Walk through the connection steps:

1. Open a terminal (or WSL terminal on Windows).
2. Run:
   ```bash
   ssh mila
   ```
3. When prompted for an OTP, open the authenticator app and enter the
   current 6-digit code. The code will **not** appear on screen as it is
   typed — this is expected.
4. On success, the Mila login-node banner appears (the ASCII art logo
   followed by system information).

Key points to communicate:
- The OTP is time-based and expires every 30 seconds; if it fails, wait
  for the next code to generate.
- The `ssh mila` shortcut works because `mila init` wrote the SSH config;
  the full hostname is `login.server.mila.quebec`.

### Step 4: Handle troubleshooting

**Not prompted to enter an OTP:**
→ `mila init` was not completed or the SSH config is missing. Run
`mila init` again (see **mila-local-setup**).

**Prompted for OTP but the login-node banner does not appear after entering
the code:**
→ MFA setup is incomplete (TOTP token missing or wrong). Return to
https://mfa.mila.quebec and add a TOTP token (see **mila-account-setup**).

**OTP rejected repeatedly:**
→ Check that the device clock is synchronized (TOTP is time-sensitive).
Contact IT support at https://it-support.mila.quebec if the issue persists.

### Step 5: Install uv on the cluster

Once the connection is confirmed, `uv` needs to be installed on the cluster
before running any jobs. Guide the user to install it now while already on the
login node.

Note: the following commands run **inside the SSH session** (on the cluster
login node) — do not offer to run them via the Bash tool.

Use the WebFetch tool to fetch **https://docs.astral.sh/uv/getting-started/installation/** for
up-to-date installation instructions. The cluster runs Linux, so follow
the Linux installation steps.

### Step 6: Point to the next skill

Once the connection is confirmed and `uv` is installed on the cluster,
point the user to **mila-run-jobs** to run their first job.
