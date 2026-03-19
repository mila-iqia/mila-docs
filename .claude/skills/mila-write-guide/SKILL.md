---
name: mila-write-guide
description: This skill should be used when the user asks to "write a guide", "create a guide", "draft a guide", "add a how-to", "write documentation for X", "create documentation for X", "add a new page about X", or wants to document a new workflow or procedure for the Mila cluster. Also use when the user asks to "modify a guide", "improve a guide", "update a guide", "edit a guide", "revise a guide", "fix a guide", or wants to make changes to an existing documentation page.
version: 1.0.0
argument-hint: <topic>
---

# Write or Edit a Guide for Mila Documentation

This skill drafts new how-to guides or improves existing ones for the Mila technical documentation site, following the MkDocs Material conventions used throughout the project.

## Workflow

### Step 0: Determine mode (new vs. existing)

If the user is editing or improving an **existing** guide, read the file first, then skip to Step 2 using the existing content as the base. Only ask clarifying questions if the requested changes are ambiguous.

If the user is writing a **new** guide, proceed with Step 1.

### Step 1: Gather requirements (interactive — new guides only)

Ask clarifying questions before drafting. Ask at most 2 questions per message to avoid overwhelming the user. Start with the most important ones:

**First message — ask:**
1. What is the goal of this guide? (What should the reader be able to do after reading it?)
2. Who is the primary audience? (e.g., new Mila users setting up their environment, experienced researchers running large jobs, ML engineers debugging distributed training)

**Second message — ask (based on first answers):**
3. What are the prerequisites? (What should the reader have already done or know?)
4. Are there related existing pages to link to as prerequisites or next steps?

**Third message — ask only if needed:**
5. Should the guide include runnable code examples? If so, what language/framework?
6. Are there any warnings, common pitfalls, or important notes to highlight?

Stop asking once enough information exists to draft a complete guide. It is fine to make reasonable assumptions for minor details and note them in the draft.

### Step 2: Draft the guide

Use the structure and MkDocs Material patterns defined in `references/guide-template.md`.

Key structural rules:
- Start with a one-paragraph introduction that states what the reader will accomplish
- Use a **Prerequisites** section with grid cards linking to prerequisite pages (if any)
- Use a **What you will do** section as a bullet list of high-level steps
- Break instructions into H2 sections, H3 subsections for sub-steps
- End with a **Next step** section using grid cards (if there is a logical next guide)
- Use a **Key concepts** section when the guide introduces new terminology

### Step 3: Standardize tone

After drafting, run the `mila-standardize-tone` skill on the draft. The content is already in context — skip Step 1 (reading the file) and run Steps 2–5 directly:
- Audit the draft for tone violations (pronouns, voice, vague language, tense, terminology)
- Present the structured violation preview
- Ask the user which categories to apply
- Apply the approved changes to the draft

Continue to Step 4 with the tone-corrected content.

### Step 4: Present the draft

Output the complete tone-corrected markdown file content in a code block. Then offer to:
- Save it directly to `docs/` with a suggested filename
- Make specific revisions before saving

## MkDocs Material Conventions

For full patterns and templates, see `references/guide-template.md`. Quick reference:

- Commands: fenced ` ```bash ` blocks
- Expected output: `<div class="result" style="border:None; padding:0" markdown>` wrapping a code block with `linenums="0"`
- Prerequisite/next-step navigation: grid cards (see template)
- Notes/warnings: `!!! note "Title"`, `!!! warning "Title"`, `!!! tip "Title"`
- Alternative implementations: `=== "Option A"` / `=== "Option B"` tabs
- Inline commands/flags: backticks
- **Mermaid diagrams**: add when a visual significantly clarifies a process —
  use `sequenceDiagram` for auth/request flows (e.g. SSH + MFA handshake),
  `flowchart` for decision logic, `graph LR` for topology. Skip diagrams when
  a numbered list is equally clear. See `references/guide-template.md` for
  examples of each type.

## Additional Resources

- **`references/guide-template.md`** — Full section templates, MkDocs Material patterns, and a complete example guide skeleton
