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

<!-- optional — at most one note, for a remark on the tool's scope (for
     example "NAS is a subset of HPO; the same technique applies"). Never a
     pointer to an example or walkthrough: those go in ## Example or
     ## Resources. -->
!!! note
    <One or two sentences.>

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

<!-- optional — any section may open with one short context paragraph:
     background the reader needs before the steps (why the tool works this
     way on the clusters, why a constraint exists, how the workflow is
     organized). Keep it to one paragraph; longer theory belongs in the
     official docs or a Technical reference page. -->
<Brief context sentence or paragraph for this section.>

<!-- Warnings about how the tool behaves on the clusters (data lost at the
     end of the job, limits shared with the job...) go as an admonition in
     the section where the reader meets the behavior. -->

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
     Sub-pages of the official docs (or of third-party docs, such as the
     NVIDIA CDI page) are linked inline in the body, where the reader
     needs them.
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
- Place each sub-section by **when the reader does it**:
    - Done **once**, per account, project, or workspace (installation,
      account setup, config files created outside jobs, editor settings such
      as the Python interpreter) → *Set up*.
    - Done in **every session or job** (connecting, launch commands, anything
      written inside the job script, including configuration generated by
      the script, such as Oríon's heredocs) → the "use" section
      (*Use in a Slurm job*, *Run on the cluster*, *Connect*).
    - A tool or account the reader must have **before the page works**
      (for example milatools installed locally) → *Before you begin*; the
      commands that configure it for this tool (for example `mila init`)
      still go under *Set up*.
- Pointers to a full walkthrough or example page (for example a
  "Complete example" tip) go in `## Example` when the category has one, or
  in `## Resources` otherwise — never as an admonition right after the
  intro.

### Containers (`containers/`)

Tools that package an environment: Singularity / Apptainer, Podman.

```markdown
## Availability            <!-- clusters, stable / tech preview, module to
                                load (if any) -->
## Get or build an image   <!-- pull from a registry, or build from a
                                definition file -->
## Run on the cluster      <!-- salloc and sbatch examples; where images and
                                data live during the job (copy the image to
                                $SLURM_TMPDIR if the tool runs from a file);
                                how to persist data (bind mounts); resource
                                limits shared by containers in a job -->
## Use GPUs                <!-- flags to expose GPUs inside the container -->
```

### Development (`development/`)

Tools to write and debug code on the clusters: VS Code, JupyterHub, Git LFS.

```markdown
## Set up                  <!-- install or configure, locally or on the cluster -->
## Connect                 <!-- main workflow, on a compute node; use
                                "Use on the cluster" if not a remote connection -->
```

Tools whose session keeps running after the browser or editor window closes
(for example JupyterHub) end *Connect* with a `### Stop the session`
sub-section: how to stop it, and what keeps running (and consuming the
allocation) otherwise.

**Install-only pages** (for example Git LFS): use `## Install` then
`## Verify the installation` instead.

### Experiments (`experiments/`)

Experiment tracking and tuning: Weights & Biases, Comet, Oríon.

```markdown
## Get access              <!-- Mila organization eligibility, access request,
                                first login; drop for tools without a Mila
                                organization -->
## Set up                  <!-- install, API key / credentials, config files
                                created once, outside jobs -->
## Use in a Slurm job      <!-- minimal job script; offline mode on compute
                                nodes without internet -->
## Example                 <!-- full example from docs/examples/ via --8<--,
                                plus the link to any step-by-step walkthrough -->
```

**First login in *Get access*.** When first login differs by account
situation (no account, existing account, account on another email domain),
choose the form that avoids repeating steps:

- **Content tabs**, one per situation, when the steps differ (tab labels
  name the situation, for example "New account" / "Existing @mila.quebec
  account").
- **One shared step list** plus a `!!! warning` for the situation that needs
  an extra step first, when the steps are mostly the same.

---

## Category index page (`<category>/index.md`)

````markdown
---
title: <Category>
description: >-
  <One sentence.>
---

# <Category>

<2–3 sentences on what this category covers. The last sentence may lead in
to the cards, for example "Docker is not available on the clusters, but
Docker images work with these tools:".>

<!-- optional — at most one admonition, for advice that applies to every
     tool in the category (for example "do not run editors on the login
     nodes", or "a virtual environment is often simpler than a
     container"). -->
!!! note
    <One or two sentences.>

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

## Top-level toolbox index (`docs/toolbox/index.md`)

The entry page of the Toolbox lists the **categories**, not the tools. Update
it only when a category is added, renamed, or removed; adding a tool never
changes it.

````markdown
---
title: Toolbox
description: >-
  <One sentence.>
---

# Toolbox

<1–2 sentences on what the Toolbox covers.>

<!-- optional — at most one admonition for a cross-category resource, such
     as the Research Project Template -->
!!! tip "<Title>"
    <One or two sentences with the link.>

<div class="grid cards" markdown>

-   [:material-<icon>:{ .lg .middle } __<Category>__](<category>/index.md)
    { .card }

    ---
    <One line: what the category covers.>

</div>
````

Rules:

- One card per category. The card title is spelled like the category
  index's `# H1`; the description matches its purpose.
- Cards follow the category order of the site nav (see *Site nav* below),
  not alphabetical order.
- No tool cards and no category sections (`## Set up`...) on this page.

---

## Site nav (`docs/README.md`)

The site navigation is defined in `docs/README.md` with the literate-nav
plugin, not in `properdocs.yml`. Every toolbox page and direct link has an
entry there, under `- [Toolbox](toolbox/index.md)`:

```markdown
- [Toolbox](toolbox/index.md)
    - [🔗 <Cross-category link>](https://<url>)
    - [<Category>](toolbox/<category>/index.md)
        - [<Tool page H1>](toolbox/<category>/<tool>.md)
        - [🔗 <External tool>](https://<tool docs or repo>)
```

Rules:

- The label of a tool page is its `# H1`, spelled exactly the same (accents
  included); the label of a category is its index's `# H1`.
- Within a category: page entries first (alphabetical), then direct-link
  entries (alphabetical), in the same order as the category index cards.
- Prefix direct-link labels with `🔗 ` so external links stand out.
- Add an entry when a page or direct link is added, fix it when a title
  changes, and remove it when a page is deleted.
- Pages pulled into another page with `include-markdown` (for example the
  Singularity sub-pages) have no nav entry.

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
- Each command runs where it belongs. When the section does not make it
  obvious, say where it runs ("on a login node", "on a compute node", "on
  the local machine"):
    - **Installation and setup** (downloading a binary, `uv add`,
      `pip install`, `wandb login`, editing `~/.bashrc`) runs on a **login
      node**. Exception: heavy installs that compile code or need a GPU (for
      example `flash-attn`) run on a compute node.
    - **Code examples and workloads** (training scripts, `podman run`,
      `orion hunt`, notebooks, editors such as VSCode) run on a **compute
      node**, never on a login node.
    - **Local commands** (for example `mila init`) run on the **local
      machine**.

### Line width

Lines are at most 80 characters (per CONTRIBUTING.md). Wrap prose, list
items, and admonition text; never wrap the exceptions below.

- **Commands** in code blocks: split long commands with a trailing `\` and a
  4-space indent on the continuation line. Split between arguments, never
  inside one.
- **Links**: a line whose length comes from a single link target may exceed
  the limit. This covers full URLs and relative paths (with or without an
  `#anchor`), in prose, in a list item, or in a grid-card title line, with
  or without trailing punctuation. Put the link on its own line to keep the
  surrounding text within the limit. Never break a link target.
- **Unsplittable arguments**: a command argument that is a single long URL
  or path (for example a `wget` download URL) may exceed the limit; do not
  break it with `\`.
- **Verbatim output and messages**: command output in result blocks
  (for example `nvidia-smi` tables) and error or log messages quoted in code
  blocks, inside admonitions included, keep their original line length.
  Never reflow them; trim repeated lines instead if they are too long to
  read.
- **Included files** (`--8<--`, `include-markdown`) are exempt.

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
- [ ] No `## Key concepts` section (new terms defined inline at first use)
- [ ] `## Resources` is the last section: 1 to 3 grid cards (not a bullet
      list), official documentation card always present (even if the
      intro links the same site), no other link already in the body, no
      doc sub-pages or general references
- [ ] Category `index.md` card matches the page (title spelled like the
      `# H1`, description matches the page's purpose); page cards first,
      then direct-link cards, each group alphabetical
- [ ] Nav entry in `docs/README.md` matches the page (label spelled like
      the `# H1`, same order as the category index cards)
- [ ] Top-level `docs/toolbox/index.md` updated if a category was added,
      renamed, or removed
- [ ] Tone rules applied (no "you", active voice, present tense, no vague words)
- [ ] No numbered headings
- [ ] No `---` between content sections — only the two bookends are permitted
      (after "Before you begin", when present, and before "Resources")
- [ ] Line width ≤ 80 characters, except the cases listed in
      *Line width* (link targets, unsplittable arguments, verbatim output
      and messages, included files)
- [ ] Included files (`--8<--`, `include-markdown`) are consistent with the
      page (same tools, same commands, e.g. `uv` vs. Conda); mismatches
      reported, not silently edited
