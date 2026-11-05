---
name: miladocs-write-toolbox
description: >-
  Use when writing, creating, drafting, editing, restructuring, or reviewing
  any page under docs/toolbox/ (containers, development, experiments tools).
  Always invoke before writing toolbox content — do not write markdown
  directly.
version: 1.0.0
argument-hint: <tool name or file-path>
---

# Write or Edit a Toolbox Page for Mila Documentation

This skill drafts new tool pages or restructures existing ones under
`docs/toolbox/`, so every tool page puts the same information in the same
place.

## Workflow

### Step 0: Determine mode and category

1. **Mode.** If the page exists, read it first and use it as the base. If the
   page is new, or exists but is too thin to fill most category sections,
   gather requirements in Step 1.
2. **Category.** Identify which section the tool belongs to. Each category has
   its own list of sections under *Category sections* in
   `references/toolbox-template.md`:
   - `containers/` — tools that package an environment
   - `development/` — tools to write and debug code on the clusters
   - `experiments/` — experiment tracking and tuning

   If the tool fits none of them, ask the user whether to create a new
   category (which needs its own `index.md`, see the template) or place it in
   the closest existing one.
3. **Page type.** Is it a category index page (`index.md`), a tool page, or
   a **direct link** (a card in the category index that points to the tool's
   own docs or repository, with no toolbox page)? Index pages use the index
   skeleton only. If the user only wants a direct link, apply the
   *Direct links to external tools* rules from the template, skip to Step 3,
   then present the index change. If unsure whether a tool deserves a page or
   just a link, apply the criteria in that template section and recommend one.

### Step 1: Gather requirements (new or thin pages)

Run this step for new pages **and** for existing pages that lack content
for most of the category sections (for example a page that only covers
login). For a thin page, skip questions the existing content already
answers, and ask about the missing sections instead of inventing content.

Ask at most 2 questions per message. Stop once enough is known to draft.

**First message — ask:**
1. What does the tool do, and why would a Mila researcher use it on the
   clusters?
2. Which clusters support it (Mila, DRAC, both), and is it stable or a tech
   preview?

**Second message — ask only what the category skeleton still needs:**
- *containers:* how images are obtained or built; GPU flags.
- *development:* local prerequisites (accounts, SSH config, local install).
- *experiments:* is there a Mila organization / access request process? Does
  the tool need internet access from compute nodes (offline mode)?
- Is there an existing example under `docs/examples/` to include?

### Step 2: Draft or restructure the page

Follow `references/toolbox-template.md`:
- Start from the **Full Toolbox Page Skeleton** (intro, optional Before you
  begin, Troubleshooting, Resources last as 1 to 3 grid cards).
  Toolbox pages have no *Key concepts* section: define new terms inline at
  first use. Include *Before you begin* only for hard prerequisites (account,
  access, local install); link background pages inline where needed.
- Fill its middle with the **category sections**, in order. Keep the
  `##` heading names verbatim so pages match across the toolbox; put
  tool-specific steps in `###` sub-sections under them, following the
  placement rules in *Category sections*. Omit a section only when the
  template says it is optional or it truly does not apply (for example
  *Get access* for tools without a Mila organization).
- When restructuring an existing page, keep all correct technical content —
  move it into the matching section rather than rewriting it. Remove exact
  duplicates (for example the same command shown twice).
- Content that fits no section: place it where it reads best and list it
  in the Step 6 report under *Content with no matching section*.
- **Included files.** Read every file pulled in with `--8<--` or
  `include-markdown`. If it contradicts the page (for example the page uses
  `uv` but the script runs `conda activate`), report the mismatch in Step 6.
  Do not edit files under `docs/examples/` unless the user asks — they are
  shared and may have generated `.diff` companions.

### Step 3: Update the category index and the nav

Always open the category's `docs/toolbox/<category>/index.md` and check the
tool's grid card against the page, not only when the title changed. The
card title must be spelled like the page's `# H1` (accents included), and
the one-line description must match the page's purpose. Add the card if the
page is new; fix it if it drifted. Page cards come first, then direct-link
cards to external tools; keep each group in alphabetical order.

Then open the site nav in `docs/README.md` (literate-nav) and check the
tool's entry under `- [Toolbox](toolbox/index.md)`, following *Site nav* in
`references/toolbox-template.md`. The label must match the `# H1`; add the
entry for a new page or direct link.

If a category was added or renamed, also update the category cards in the
top-level `docs/toolbox/index.md` (see *Top-level toolbox index* in the
template).

### Step 4: Standardize format

Apply the `miladocs-standardize-format` steps **inline** on the draft — do
NOT invoke it via the Skill tool (that would start a fresh context and lose
the draft). Skip its Step 0 and run Steps 1–4 directly. Toolbox pages are
exempt from the guide-only checks (*What this guide covers*, *Next step*);
they end with `## Resources` instead, have no *Key concepts* section, and
`---` bookends sit after *Before you begin* (only when present) and before
*Resources*. Skip
the format skill's **Naming** category (its `Userguide_` / `Information_`
convention does not apply here); check filenames against *Naming Conventions* in
`references/toolbox-template.md` instead.

### Step 5: Standardize tone

Apply the `miladocs-standardize-tone` steps **inline** on the
format-corrected draft — do NOT invoke it via the Skill tool. Skip its
Step 1 and run Steps 2–5 directly.

**Apply mode.** If the user asked to *apply* the skill to a page (rather than
to draft or review it), apply all format and tone fixes in Steps 4–5
without the per-category approval prompts, and save directly in Step 6.

### Step 6: Present the result

**Draft mode** (default): output the complete page (and the index and nav
diffs, if any) in code blocks. Then offer to:
- Save it to `docs/toolbox/<category>/<tool>.md`
  (`lowercase_with_underscores.md`) or overwrite the existing file
- Make specific revisions before saving

Then report, under these headings (omit a heading with nothing to list):

- **Changes**: what moved where, and what was added or removed.
- **Missing sections**: category sections left out because the page has no
  content for them, and the questions that would fill them.
- **Content added or inferred (to verify)**: anything not taken from the
  original page or an included file.
- **Mismatches in included files**: contradictions between the page and
  files pulled in with `--8<--` or `include-markdown`.
- **Content with no matching section**: where it was placed, and why.

## Additional Resources

- **`references/toolbox-template.md`** — Full page skeleton, category
  sections, category and top-level index pages, direct links, site nav,
  naming, style rules (including line width), and pre-save checklist
- **`../shared/miladocs-mkdocs-patterns.md`** — Frontmatter and MkDocs
  Material patterns (admonitions, tabs, grid cards, includes)
