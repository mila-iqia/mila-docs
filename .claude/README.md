# Claude Code — Mila Docs Skills

This directory contains Claude Code configuration for the [Mila documentation](https://docs.mila.quebec/) repository. The `skills/` subdirectory defines custom workflows ("skills") that help cluster users get answers about setting up and using the Mila cluster.

## Skills overview

| Skill | Trigger phrases | Description |
|-------|----------------|-------------|
| `mila-account-setup` | "How do I get a Mila account", "How do I set up MFA", "I received a registration token" | Guides through obtaining a Mila account, enabling cluster access, and setting up MFA |
| `mila-local-setup` | "How do I install WSL", "How do I install uv", "How do I install milatools", "mila init" | Guides through setting up WSL (Windows), uv, and milatools on a local machine |
| `mila-connect-cluster` | "How do I connect to the cluster", "How do I SSH", "I can't connect", "OTP" | Guides through SSH connection, OTP entry, and connection troubleshooting |
| `mila-run-jobs` | "How do I run a job", "mila code", "sbatch", "How do I use a GPU", "train a model" | Guides through interactive development with `mila code` and batch job submission with `sbatch` |

## Usage

Start a Claude Code session at the repo root, then describe what you want in plain language:

```
How do I set up MFA?
```

```
How do I install milatools on Windows?
```

```
How do I submit a batch job with sbatch?
```

Claude will detect the matching skill and follow its workflow automatically.

## Contributing a new skill

Ask Claude to build it for you — describe the workflow you want and it will create the necessary files. Name the skill `mila-<name>` — the `.gitignore` in this directory only tracks `skills/mila-*/`, so skills with any other name will not be committed.
