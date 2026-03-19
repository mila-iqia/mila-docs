---
name: miladocs-standardize-tone
description: This skill should be used when the user asks to "standardize the tone", "fix the tone", "rewrite this page", "improve tone consistency", "apply tone rules", "standardize this documentation", or mentions tone, voice, or pronoun issues in a documentation page. Also applies when the user says a page uses too much "you" or second-person language.
version: 1.0.0
argument-hint: <path-to-doc>
---

# Standardize Tone for Mila Documentation

This skill audits a Mila documentation page for tone violations and proposes corrections. It covers pronoun usage, voice, vague language, verb tense, and Mila-specific terminology.

## Workflow

### Step 1: Read the target file

If the user provided a file path as an argument, read that file. Otherwise, ask which file to standardize before proceeding.

### Step 2: Audit for tone violations

Scan the file for each rule category defined in `references/tone-rules.md`. For each violation found, record:
- The original sentence or phrase
- The rule it violates
- A proposed replacement

Group findings by rule category.

### Step 3: Present a preview

Show the user a structured preview of proposed changes. Format as:

```
## Tone Audit: <filename>

### Second-person pronouns (X instances)
- Line N: "You can run the script" → "Run the script" (or "The script can be run")
- ...

### Passive voice (X instances)
- Line N: "The job should be submitted" → "Submit the job"
- ...

### Vague language (X instances)
- Line N: "Simply run the command" → "Run the command"
- ...

### Tense (X instances)
- ...

### Terminology (X instances)
- ...
```

If a category has no violations, omit it from the preview.

### Step 4: Ask for confirmation

After the preview, ask:
> Apply all changes? Or specify which categories to apply (pronouns / voice / vague / tense / terminology), or type "cancel".

### Step 5: Apply approved changes

Edit the file applying only the approved changes. Prefer minimal edits — change only the flagged phrases, not surrounding content.

## Rules Summary

Full rules with examples are in `references/tone-rules.md`. Summary:

| Rule | Replace | With |
|------|---------|------|
| Third-person | "you", "your" | restructure or use "the user" |
| Active voice | passive constructions | active verb phrases |
| No vague language | simply, just, easy, obviously, etc. | remove or rephrase |
| Present tense | "you will X", "you'll X" | imperative "X" |
| Mila terminology | server, worker, task (SLURM) | cluster, compute node, job |

## Important Constraints

- Do not change code blocks, command examples, or expected output blocks.
- Do not change file names, SLURM flags, or other technical identifiers.
- Do not alter the meaning of sentences — if a safe third-person rewrite is not obvious, flag it for human review rather than guessing.
- Do not change admonition titles or MkDocs directives.
- Preserve all markdown formatting (bold, links, admonitions, tabs, etc.).

## Additional Resources

- **`references/tone-rules.md`** — Complete rules with before/after examples for each category
