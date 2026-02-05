---
name: rst-to-md-mkdocs-one-page
description: Converts a single reStructuredText (.rst) page to MkDocs-compatible Markdown (.md) using the project converter, then applies post-conversion fixes (headings, lists, internal links). Use when the user wants one RST file converted to MD and ready for MkDocs, or when converting .rst to .md with proper headings and links.
---

# Convert One RST Page to MkDocs-Ready Markdown

## When to use

- User asks to convert a specific `.rst` file to `.md` so it is **compatible with MkDocs** or **ready for MkDocs**
- User wants one RST page converted and expects correct headings, numbered lists, and internal links
- Same triggers as single-page RST→MD conversion, with emphasis on MkDocs compatibility

## Workflow

### 1. Run the project converter

From the **project root**, run the converter for the single RST file:

```bash
python -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from scripts.rst_to_mkdocs_md import rst_to_md

rst_path = Path('docs/YOUR_PAGE.rst')   # replace with actual path
docs_root = Path('docs')
text = rst_path.read_text(encoding='utf-8')
md_content = rst_to_md(text, rst_path=rst_path, docs_root=docs_root)
md_path = rst_path.with_suffix('.md')
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text(md_content, encoding='utf-8')
print(f'Converted: {rst_path} -> {md_path}')
"
```

Replace `docs/YOUR_PAGE.rst` with the actual relative path (e.g. `docs/Information_roles_and_resources.rst`).

### 2. Apply MkDocs post-conversion fixes

Open the generated `.md` and fix the following so the page renders correctly in MkDocs:

**Headings**

- The converter may leave section titles as plain lines. Turn them into Markdown headings:
  - RST `####` with overline for first-level sections → `# Part`
  - RST `****` with overline for second-level sections → `## Chapter`
  - RST `====` for third-level sections → `### Section`
  - RST `----` for forth-level sections → `#### Subsection`
  - RST `^^^^` for fifth-level sections → `##### Subsubsection`
  - RST `""""` for sixth-level sections → `###### Paragraph`

**Numbered lists**

- RST auto-numbered items use `#.`. In Markdown that can be mistaken for a heading. Replace with explicit numbers:
  - `#. First item` → `1. First item`
  - `#. Second item` → `2. Second item`
  - (and so on for the rest of the list).

**Internal links**

- `:ref:` targets often become `[Label](#refname)`. In MkDocs, anchors are derived from headings; `#refname` may not exist.
- Replace with the correct internal link:
  - Prefer a relative path to the target doc, e.g. `[SLURM](Userguide_running_code.md)`.
  - Use the nav in `docs/README.md` (or the main nav file) to find the right `.md` file for the topic.
  - If the target page has a specific section, use `[Label](OtherPage.md#section-anchor)` (MkDocs generates anchors from heading text).

### 3. Optional

- Add the new page to the nav in `docs/README.md` (or the project’s nav file) if it is not already listed.
- Do not modify or delete the original `.rst`.

## Notes

- **Output**: The `.md` file is written next to the `.rst` (same directory, same base name, `.md` extension).
- **Docs root**: Always use `docs_root=Path('docs')` when calling `rst_to_md` so `:doc:` / `:ref:` resolve correctly.
- **Batch**: For converting many files at once, run `python scripts/rst_to_mkdocs_md.py`; this skill is for **one page at a time** with MkDocs-ready fixes.
