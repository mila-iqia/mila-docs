# Claude Code — Mila Docs Skills

This directory contains Claude Code configuration for the [Mila documentation](https://docs.mila.quebec/) repository. The `skills/` subdirectory defines custom workflows ("skills") that contributors can invoke to write or improve documentation pages.

## Skills overview

| Skill | Trigger phrases | Description |
|-------|----------------|-------------|
| `miladocs-write-guide` | "write a guide for X", "create a how-to for X", "update the guide at docs/X.md" | Drafts or edits a how-to guide following the guide template and MkDocs Material conventions |
| `miladocs-standardize-tone` | "standardize the tone of docs/X.md", "fix the tone", "apply tone rules to X" | Audits a doc for tone violations and applies fixes |

## Usage

Start a Claude Code session at the repo root, then describe what you want in plain language:

```
write a guide for setting up VS Code Remote on the Mila cluster
```

```
standardize the tone of docs/Userguide_login_mfa.md
```

Claude will detect the matching skill and follow its workflow automatically. For writing guides, it will ask a few clarifying questions before drafting. For tone standardization, it will show a preview of proposed changes before editing the file.

## Skill details

### `miladocs-write-guide`

Drafts new how-to guides or improves existing ones. For new guides, Claude asks about the goal, audience, prerequisites, and whether to include code examples. It then produces a complete markdown file following `skills/miladocs-write-guide/references/guide-template.md`, and runs the tone standardization step automatically before presenting the draft.

### `miladocs-standardize-tone`

Audits a documentation page against the rules in `skills/miladocs-standardize-tone/references/tone-rules.md`. It checks for second-person pronouns, passive voice, vague language (simply, just, easy…), future tense, and incorrect Mila terminology. Changes are shown as a categorized preview and applied only after confirmation.

## Contributing a new skill

Ask Claude to build it for you — describe the workflow you want and it will create the necessary files. Name the skill `miladocs-<name>` — the `.gitignore` in this directory only tracks `skills/miladocs-*/`, so skills with any other name will not be committed.
