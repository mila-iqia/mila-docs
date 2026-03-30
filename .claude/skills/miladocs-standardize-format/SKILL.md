---
name: miladocs-standardize-format
description: Use when writing, editing, updating, or improving any Mila documentation page, or when the user asks about formatting, structure, or doc standards.
version: 1.0.0
argument-hint: <file-path or topic>
---

# Standardize Format for Mila Documentation

This skill audits a documentation page against the MkDocs Material conventions
defined in `../shared/miladocs-mkdocs-patterns.md` and applies approved fixes
interactively.

## Workflow

### Step 0: Load content

If the file path is known, read the file. If content is already in context
(e.g. called from `miladocs-write-guide`), skip the read and use that content
directly.

### Step 1: Audit for violations

Check the content against `../shared/miladocs-mkdocs-patterns.md`. Group
violations into these categories:

- **Frontmatter** — missing `title` or `description` field, or frontmatter
  absent entirely
- **Naming** — filename does not follow `Userguide_<topic>.md` or
  `Information_<topic>.md` convention (flag only; cannot rename automatically)
- **Headings** — numbered headings (e.g. `### 2. First-time login`), wrong
  case (title case on section headings, sentence case on page title)
- **Code blocks** — shell commands not in ` ```bash ` blocks, expected output
  not wrapped in `<div class="result" style="border:None; padding:0" markdown>`
- **Admonitions** — wrong syntax, missing 4-space indent on content
- **Grid cards** — navigation cards (Before you begin / Next step) not using
  the standard grid card pattern; single card missing `&nbsp;` spacer
- **Separators** — `---` used between regular content sections (should only
  appear before `## Key concepts` and `## Next step / Next steps`)

### Step 2: Present violations

For each category that has violations, show:

```
**<Category>** (<N> issue(s))
  - Line ~X: <short description of the problem>
  - Line ~Y: <short description of the problem>
```

If no violations are found in a category, omit it. If the page is clean,
report that and stop.

### Step 3: Ask which categories to fix

Ask the user which violation categories to apply. Present them as a
checklist. The user may select all, some, or none.

### Step 4: Apply approved fixes

Apply only the approved categories. Output the corrected file content in a
full markdown code block.

Then offer to:
- Save the file (using the original path, or a suggested path if new)
- Make additional revisions before saving

## Notes

- **Naming violations** can only be flagged — suggest the correct filename
  but do not rename the file automatically. Ask the user to confirm before
  suggesting a `git mv` command.
- This skill focuses on structure and markup. For prose tone and voice, use
  `miladocs-standardize-tone` instead.
- When called from `miladocs-write-guide`, content is already in context —
  skip Step 0 and run Steps 1–4 directly on the draft.
