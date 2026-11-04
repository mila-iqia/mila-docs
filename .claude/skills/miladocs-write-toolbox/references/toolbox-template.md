# Toolbox Template for Mila Documentation

Reference templates and MkDocs Material patterns. Use these when drafting a
new toolbox page or restructuring an existing one under `docs/toolbox/`.

For frontmatter requirements, MkDocs Material patterns, and naming conventions,
see `../../shared/miladocs-mkdocs-patterns.md`.

A toolbox page documents **one tool**; a guide teaches **one task**. Toolbox
pages reuse the guide patterns (Before you begin, result blocks, tabs,
admonitions) but replace "What this guide covers" and "Next step" with
category sections and a closing **Resources** section.

---

## Full Toolbox Page Skeleton

````markdown
# <Title: tool name, e.g. "Weights & Biases" or "Podman">

<Intro paragraph, 2–3 sentences. State what the tool is and why a Mila
researcher would use it on the clusters. Link to the official site once, here.
Do not use "you" — use imperative mood or third-person.>

<!-- optional — include only for hard prerequisites: something the reader
     must have done or obtained before the page works (an account, access to
     an organization, a local install, SSH set up). Merely useful background
     is linked inline where it is needed, not here. When the section is omitted, also omit the "---" after it. -->
## Before you begin

<!-- Pattern A — link to prerequisite Mila pages (use grid cards). -->
<div class="grid cards" markdown>

-   [:material-<icon>:{ .lg .middle } __<Prerequisite page title>__](<file>.md)
    { .card }

    ---
    <One-line description of what the prerequisite page covers.>

</div>

<!-- Pattern B — accounts, local installs, or access the tool requires: -->
!!! success "<Tool or account requirements>"
    - <Requirement> <One-line description of the requirement.>

<!-- Both patterns can appear together. -->

---

<!-- Category sections go here, in the order given in "Category sections"
     below. Keep the heading names verbatim so pages match across the
     toolbox. -->

## <Category section 1>

<Brief context sentence for this section.>

```bash
<command>
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
<expected output>
```
</div>

<!-- Cluster differences (Mila vs. DRAC) go in content tabs, not separate
     sections: -->

=== "Mila cluster"

    ```bash
    <command on the Mila cluster>
    ```

=== "DRAC clusters"

    ```bash
    <command on DRAC clusters>
    ```

<!-- Shared example files are included, never copied: -->
```python
--8<-- "docs/examples/<path>"
```

## <Category section 2>

...

## Troubleshooting

<!-- optional — one collapsible admonition per known issue; omit the section
     entirely if there are no known issues -->
??? question "<Symptom or error message>"
    <Cause and fix.>

---

<!-- optional — include only when new terms are introduced -->
## Key concepts

`<term>`
:   <Definition.>

## Resources

<!-- Always the last section. Grid cards, never a bullet list. Keep it
     short: 1 to 3 cards. Include only:
       - the tool's official documentation (always, even when the intro
         already links the same site)
       - its source repository, only when Mila maintains the tool
       - a Mila page that goes deeper on this tool and is not linked in
         the body
     Leave out: sub-pages of the official docs (SDK/API reference, a
     specific guide), general references (Slurm, Python), alternative
     tools, and anything else already linked in the page body. The
     official-documentation card is the one exception to that rule.
     Icons:
       official documentation   :material-book-open-variant:
       source repository        :simple-github:
       related Mila page        icon matching the page's topic
     A single card needs the &nbsp; spacer before </div>. -->
<div class="grid cards" markdown>

-   [:material-book-open-variant:{ .lg .middle } __<Tool> documentation__](<url>)
    { .card }

    ---
    <One line: what the reader finds there.>

-   [:material-<icon>:{ .lg .middle } __<Related Mila page title>__](<relative path>.md)
    { .card }

    ---
    <One line: what the page covers.>

</div>
````

---

## Category sections

Place these between the first `---` and `## Troubleshooting`, in this order.

- These `##` headings are the **required top level**: keep their names
  verbatim. Steps specific to one tool (for example *Configure Oríon*,
  *Define the search space*) go as `###` sub-sections under the closest
  `##` section, not as new `##` sections.
- Place each sub-section by **when the reader does it**: one-time
  installation and account setup go under *Set up*; anything written or run
  inside every job (job-script configuration, launch commands) goes under the
  "use" section (*Use in a Slurm job*, *Run on the cluster*, *Connect*).
- Pointers to a full walkthrough or example page (for example a
  "Complete example" tip) go in `## Example` when the category has one, or
  in `## Resources` otherwise — never as an admonition right after the
  intro.

### Containers (`containers/`)

Tools that package an environment: Singularity / Apptainer, Podman.

```markdown
## Availability            <!-- clusters, stable / tech preview, module to load -->
## Get or build an image   <!-- pull from a registry, or build from a definition file -->
## Run on the cluster      <!-- salloc and sbatch examples; copy image to $SLURM_TMPDIR, bind data -->
## Use GPUs                <!-- flags to expose GPUs inside the container -->
```

### Development (`development/`)

Tools to write and debug code on the clusters: VS Code, JupyterHub, Git LFS.

```markdown
## Set up                  <!-- install or configure, locally or on the cluster -->
## Connect                 <!-- main workflow, on a compute node; use
                                "Use on the cluster" if not a remote connection -->
```

**Install-only pages** (for example Git LFS): use `## Install` then
`## Verify the installation` instead.

### Experiments (`experiments/`)

Experiment tracking and tuning: Weights & Biases, Comet, Oríon.

```markdown
## Get access              <!-- Mila organization eligibility, access request,
                                first login (new vs. existing account in tabs);
                                drop for tools without a Mila organization -->
## Set up                  <!-- install, API key / credentials, config location -->
## Use in a Slurm job      <!-- minimal job script; offline mode on compute
                                nodes without internet -->
## Example                 <!-- full example from docs/examples/ via --8<--,
                                plus the link to any step-by-step walkthrough -->
```

---

## Category index page (`<category>/index.md`)

````markdown
---
title: <Category>
description: >-
  <One sentence.>
---

# <Category>

<2–3 sentences on what this category covers.>

<div class="grid cards" markdown>

-   [:material-<icon>:{ .lg .middle } __<Tool>__](<tool>.md)
    { .card }

    ---
    <One line: what the tool is for.>

-   [:material-<icon>:{ .lg .middle } __<External tool>__](https://<tool docs or repo>)
    { .card }

    ---
    <One line: what the tool is for.>

</div>
````

### Direct links to external tools

A category can list a tool **without a toolbox page**, with a card that links
straight to the tool's own documentation or repository (for example `cluv`,
`milatools`, `milalib`). Use a direct link when:

- the tool already has its own maintained documentation (often a Mila-built
  tool), and
- there is no cluster-specific setup worth documenting here.

Write a toolbox page instead when using the tool on the clusters needs
specific steps (access, modules, compute-node constraints, offline mode...).

Rules for direct-link cards:

- Link to the tool's documentation site when one exists, otherwise to its
  repository. Use the full `https://` URL.
- Same card format as page cards.
- Order: page cards first (alphabetical), then direct-link cards
  (alphabetical).
- Do not create an empty or stub `.md` page just to hold a link.

---

## Naming Conventions

- Filename: `lowercase_with_underscores.md`, placed in
  `docs/toolbox/<category>/` (`containers/`, `development/`, `experiments/`).
- Page title: the tool's official name and casing ("Oríon", "Git LFS",
  "Weights & Biases").
- Sentence case for section headings: "Get or build an image".
- No numbers in section headings.
- One page per tool.

---

## Writing style rules

Same rules as `../../miladocs-write-guide/references/guide-template.md`
(complete clauses in alternatives), plus:

- Notes and warnings use admonitions, never `> **NOTE**` blockquotes.
- Every example runs on a compute node, never on a login node.

---

## Checklist Before Saving

- [ ] YAML frontmatter with `title` and one-sentence `description`
- [ ] Single `# H1` matching `title`
- [ ] Intro paragraph (2–3 sentences, no "you/your") with one link to the
      official site
- [ ] Before you begin section only for hard prerequisites — grid cards
      (Pattern A) and/or admonition (Pattern B); otherwise omitted, with
      background pages linked inline
- [ ] Category sections present, in order, with the template's heading names
- [ ] All commands in fenced blocks with a language
- [ ] Expected output in `<div class="result">` blocks
- [ ] Examples included with `--8<--`, not copied
- [ ] Mila vs. DRAC differences in content tabs
- [ ] Troubleshooting items as `??? question` admonitions (or section omitted)
- [ ] Key concepts as definition list (only if new terms introduced)
- [ ] `## Resources` is the last section: 1 to 3 grid cards (not a bullet
      list), official documentation card always present (even if the
      intro links the same site), no other link already in the body, no
      doc sub-pages or general references
- [ ] Category `index.md` card matches the page (title spelled like the
      `# H1`, description matches the page's purpose); page cards first,
      then direct-link cards, each group alphabetical
- [ ] Tone rules applied (no "you", active voice, present tense, no vague words)
- [ ] No numbered headings
- [ ] No `---` between content sections — only the two bookends are permitted
      (after "Before you begin", when present, and before "Key concepts" /
      "Resources")
- [ ] Line width ≤ 80 characters (per CONTRIBUTING.md):
      - Long shell commands in code blocks are split with a trailing `\`
        and a 4-space indent on the continuation line.
      - A line whose length comes from a single URL may exceed the limit
        (a link in prose, a list item, or a grid-card title line, with or
        without trailing punctuation). Never break a URL across lines.
      - Included example files (`--8<--`) are exempt.
- [ ] Included files (`--8<--`, `include-markdown`) are consistent with the
      page (same tools, same commands, e.g. `uv` vs. Conda); mismatches
      reported, not silently edited
