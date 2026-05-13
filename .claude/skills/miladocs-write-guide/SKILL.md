---
name: miladocs-write-guide
description: >-
  Use when writing, creating, drafting, or editing any user guide or how-to page
  in the Mila documentation. Always invoke before writing any doc content — do
  not write markdown directly.
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
- Use a **What this guide covers** section as a bullet list of high-level steps
- Break instructions into H2 sections, H3 subsections for sub-steps
- End with a **Next step** section using grid cards (if there is a logical next guide)
- Use a **Key concepts** section when the guide introduces new terminology

### Step 3: Standardize format

Apply the `miladocs-standardize-format` steps **inline** on the draft — do
NOT invoke it via the Skill tool (that would start a fresh context and lose
the draft). The content is already in context — skip Step 0 and run Steps
1–4 directly:
- Audit for violations (frontmatter, headings, code blocks, admonitions,
  grid cards, separators)
- Present the structured violation preview
- Ask the user which categories to apply
- Apply the approved changes to the draft

Continue to Step 4 with the format-corrected content.

### Step 4: Standardize tone

Apply the `miladocs-standardize-tone` steps **inline** on the
format-corrected draft — do NOT invoke it via the Skill tool. The content
is already in context — skip Step 1 (reading the file) and run Steps 2–5
directly:
- Audit the draft for tone violations (pronouns, voice, vague language,
  tense, terminology)
- Present the structured violation preview
- Ask the user which categories to apply
- Apply the approved changes to the draft

Continue to Step 5 with the tone-corrected content.

### Step 5: Present the draft

Output the complete tone-corrected markdown file content in a code block. Then offer to:
- Save it to `docs/` — ask the user which subdirectory is most appropriate
  (`getting_started/`, `userguides/`, `toolbox/`, `technical_reference/`,
  `help/`, or other), then suggest a `lowercase_with_underscores.md` filename
- Make specific revisions before saving

## Additional Resources

- **`references/guide-template.md`** — Full guide skeleton, writing style rules,
  and pre-save checklist
- **`../shared/miladocs-mkdocs-patterns.md`** — Frontmatter, naming conventions,
  and all MkDocs Material patterns (admonitions, code blocks, grid cards,
  Mermaid diagrams, etc.)
