#!/usr/bin/env python3
"""Convert Sphinx/ReST docs to MkDocs-flavored Markdown.

This script is intentionally "best effort" and designed for a *transition* period:
- it does not delete or modify any .rst files
- it creates (or overwrites) adjacent .md files next to each .rst file

It handles the common constructs used in this repository:
- underline-style headings
- .. code-block:: and .. prompt::
- admonitions: note/warning/tip/important
- .. image::
- .. include:: (converted to an HTML comment placeholder)
- .. toctree:: (converted to a bullet list)
- basic :ref: and :doc: roles

Run:
    python scripts/rst_to_mkdocs_md.py

Optional args:
    --docs-dir docs
    --dry-run
    --keep-existing  (don't overwrite existing .md)

"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def rst_to_md(text: str, *, rst_path: Path, docs_root: Path) -> str:
    """Convert one RST document (string) to MkDocs-friendly Markdown.

    Args:
        text: Raw RST contents.
        rst_path: Path to the source .rst file.
        docs_root: Root docs directory (used to compute relative links).
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n")

    # Capture Sphinx-only explicit targets like: .. _label:
    # We'll keep them as explicit HTML anchors so existing :ref: links remain valid.
    def repl_label(m: re.Match) -> str:
        label = m.group(1).strip()
        return f'<a id="{label}"></a>\n\n'

    text = re.sub(r"^\.\. _([^:]+):\s*\n\s*\n", repl_label, text, flags=re.M)

    # Convert raw HTML blocks into fenced HTML
    def repl_raw(m: re.Match) -> str:
        raw = m.group(1)
        raw = re.sub(r"^ {3}", "", raw, flags=re.M)
        return f"```html\n{raw.strip()}\n```\n\n"

    text = re.sub(r"^\.\. raw:: html\n\n((?: {3}.*\n)+)\n?", repl_raw, text, flags=re.M)

    # Drop .. meta:: blocks (MkDocs has meta via YAML frontmatter; we can add later)
    text = re.sub(r"^\.\. meta::\n(?:\s+:.*\n)+\n?", "", text, flags=re.M)

    # Include directives -> placeholder comment (handled later by a plugin or manual inlining)
    def repl_include(m: re.Match) -> str:
        inc = m.group(1).strip()
        inc_md = re.sub(r"\.rst$", ".md", inc)
        return f"<!-- include: {inc_md} -->\n\n"

    text = re.sub(r"^\.\. include::\s+(.+?)\s*\n\s*\n", repl_include, text, flags=re.M)

    # Toctree -> bullet list
    def repl_toctree(m: re.Match) -> str:
        body = m.group(1)
        items: list[str] = []
        for line in body.splitlines():
            if not line.strip():
                continue
            if line.strip().startswith(":"):
                continue
            entry = line.strip()

            # Skip glob patterns (Sphinx :glob:) since they don't point to a single file.
            if "*" in entry or "?" in entry or "[" in entry:
                continue

            # External link with explicit title: Title <url>
            m2 = re.match(r"(.+?)\s*<\s*(https?://[^>]+)\s*>", entry)
            if m2:
                title, url = m2.group(1).strip(), m2.group(2).strip()
                items.append(f"- [{title}]({url})")
                continue

            # Local doc path (typically extensionless). Interpret it relative to the
            # current document directory (Sphinx toctree behavior).
            entry_path = Path(entry)
            if entry_path.suffix in {".rst", ".md"}:
                entry_path = entry_path.with_suffix("")

            # Prefer linking to an index page when the entry refers to a folder name.
            # (Many of our docs use foo/index in toctrees already, so this is mostly
            # harmless.)
            target_rel_to_docs = (
                rst_path.relative_to(docs_root).parent / entry_path
            ).with_suffix(".md")
            cur_md_rel_to_docs = rst_path.relative_to(docs_root).with_suffix(".md")
            rel_link = Path(
                os.path.relpath(target_rel_to_docs, start=cur_md_rel_to_docs.parent)
            )
            items.append(f"- [{entry_path.name}]({rel_link.as_posix()})")

        if not items:
            return ""
        return "\n".join(items) + "\n\n"

    text = re.sub(
        r"^\.\. toctree::\n(?:(?:\s+:.*\n)+)?\n((?:\s{3,}.*\n)+)\n?",
        repl_toctree,
        text,
        flags=re.M,
    )

    # Headings: underline style to Markdown ATX style (must run before removing underlines)
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            ul = lines[i + 1]
            if re.fullmatch(r"[#=\-\^\*\+~]{3,}", ul.strip()):
                ch = ul.strip()[0]
                level_map = {
                    "#": 1,
                    "*": 2,
                    "=": 3,
                    "-": 4,
                    "^": 5,
                    '"': 6,
                }
                lvl = level_map.get(ch, 2)
                out.append("#" * lvl + " " + line.strip())
                out.append("")
                i += 2
                continue
        out.append(line)
        i += 1
    text = "\n".join(out) + "\n"

    # Remove stray underline-only lines that can remain (e.g. from tables).
    text = re.sub(r"^=+\s*$", "", text, flags=re.M)
    text = re.sub(r"^-+\s*$", "", text, flags=re.M)

    # Convert literalinclude directives into a fenced block placeholder.
    # We don't inline files here (some are large or not meant to render), but this
    # eliminates warnings about unknown directives.
    def repl_literalinclude(m: re.Match) -> str:
        path = m.group(1).strip()
        lang = m.group(2)
        lang = (lang or "").strip()
        fence = lang if lang else "text"
        return f"```{fence}\n# (literalinclude) {path}\n```\n\n"

    text = re.sub(
        r"^\.\. literalinclude::\s+(.+?)\s*\n(?:\s+:language:\s*([^\n]+)\n)?\s*\n",
        repl_literalinclude,
        text,
        flags=re.M,
    )

    # Code blocks: .. code-block:: lang
    def repl_codeblock(m: re.Match) -> str:
        lang = m.group(1).strip()
        code = m.group(2)
        code = re.sub(r"^ {3}", "", code, flags=re.M)
        return f"```{lang}\n{code.rstrip()}\n```\n\n"

    text = re.sub(
        r"^\.\. code-block::\s*(\w+)\s*\n\n((?: {3}.*\n)+)\n?",
        repl_codeblock,
        text,
        flags=re.M,
    )

    # Prompt blocks: .. prompt:: bash $
    def repl_prompt(m: re.Match) -> str:
        prompt = m.group(1).strip()
        body = m.group(2)
        body = re.sub(r"^ {3}", "", body, flags=re.M)
        return f"```bash\n# {prompt}\n{body.rstrip()}\n```\n\n"

    text = re.sub(
        r"^\.\. prompt::\s*(.+?)\s*\n\n((?: {3}.*\n)+)\n?",
        repl_prompt,
        text,
        flags=re.M,
    )

    # Admonitions: .. note::, .. warning::, .. tip::, .. important::
    def repl_admon(m: re.Match) -> str:
        kind = m.group(1).strip().lower()
        title = m.group(2)
        body = m.group(3)
        body = re.sub(r"^ {3}", "", body, flags=re.M).rstrip("\n")

        kind_map = {
            "note": "note",
            "warning": "warning",
            "tip": "tip",
            "important": "important",
        }
        mk = kind_map.get(kind, "note")

        if title and title.strip():
            t = title.strip().replace('"', '\\"')
            return f'!!! {mk} "{t}"\n\n    ' + "\n    ".join(body.splitlines()) + "\n\n"

        return f"!!! {mk}\n\n    " + "\n    ".join(body.splitlines()) + "\n\n"

    text = re.sub(
        r"^\.\. (note|warning|tip|important)::\s*(.*?)\n\n((?:(?: {3,}.*)\n)+)\n?",
        repl_admon,
        text,
        flags=re.M,
    )

    # Images: .. image:: path
    def repl_img(m: re.Match) -> str:
        path = m.group(1).strip()
        return f"![]({path})\n\n"

    text = re.sub(r"^\.\. image::\s+(.+?)\s*\n\s*\n", repl_img, text, flags=re.M)

    # Inline roles
    # :ref:`Text <label>` or :ref:`label`
    # For explicit labels, keep them verbatim (but normalize to lowercase) so they
    # match the <a id="label"> anchors we inject from '.. _label:' targets.
    text = re.sub(
        r":ref:`([^`<]+?)\s*<([^`>]+)>`",
        lambda m: f"[{m.group(1)}](#{m.group(2).strip().lower()})",
        text,
    )

    def _anchor_from_text(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9\s\-]", "", s)
        # Use '_' to better match existing Sphinx label conventions in this repo.
        s = re.sub(r"\s+", "_", s)
        return s

    text = re.sub(
        r":ref:`([^`]+?)`",
        lambda m: f"[{m.group(1)}](#{_anchor_from_text(m.group(1))})",
        text,
    )

    def _doc_link(target: str) -> str:
        # target can be absolute like /examples/foo/index or relative (to current doc).
        target = target.strip()
        if target.startswith("/"):
            target_rel_to_docs = Path(target.lstrip("/"))
        else:
            cur_rel_dir = rst_path.relative_to(docs_root).parent
            target_rel_to_docs = cur_rel_dir / Path(target)

        # Drop any explicit extension the author might have used.
        if target_rel_to_docs.suffix in {".rst", ".md"}:
            target_rel_to_docs = target_rel_to_docs.with_suffix("")

        target_md_rel_to_docs = target_rel_to_docs.with_suffix(".md")
        cur_md_rel_to_docs = rst_path.relative_to(docs_root).with_suffix(".md")
        rel = os.path.relpath(target_md_rel_to_docs, start=cur_md_rel_to_docs.parent)
        return Path(rel).as_posix()

    # :doc:`Title </path/to/doc>`
    text = re.sub(
        r":doc:`([^`<]+?)\s*<\s*([^`>]+?)\s*>`",
        lambda m: f"[{m.group(1).strip()}]({_doc_link(m.group(2))})",
        text,
    )
    # :doc:`/path/to/doc`
    text = re.sub(
        r":doc:`\s*([^`]+?)\s*`",
        lambda m: f"[{Path(m.group(1).strip()).name}]({_doc_link(m.group(1))})",
        text,
    )

    # RST external links: `text <url>`_ and bare <url>_
    text = re.sub(r"`([^`]+?)\s*<\s*(https?://[^>]+)\s*>`_", r"[\1](\2)", text)
    text = re.sub(r"<\s*(https?://[^>]+)\s*>`?_", r"[\1](\1)", text)

    # Fix common bullet links we generate from toctree/prereq lists that end up being
    # relative-to-current-doc but were emitted as docs-root relative.
    # This is a conservative pass: it only rewrites links that start with 'examples/'.
    def repl_examples_link(m: re.Match) -> str:
        target = m.group(1)
        cur_md_rel_to_docs = rst_path.relative_to(docs_root).with_suffix(".md")
        rel = os.path.relpath(Path(target), start=cur_md_rel_to_docs.parent)
        return f"]({Path(rel).as_posix()})"

    text = re.sub(r"\]\((examples/[^)]+)\)", repl_examples_link, text)

    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir", default="docs", help="Docs directory (default: docs)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Don't overwrite existing .md files",
    )
    args = parser.parse_args()

    root = Path(args.docs_dir)
    if not root.exists():
        raise SystemExit(f"Docs dir not found: {root}")

    rst_files = [p for p in root.rglob("*.rst") if p.is_file()]

    created = 0
    skipped = 0
    for rst in rst_files:
        md = rst.with_suffix(".md")
        if args.keep_existing and md.exists():
            skipped += 1
            continue

        src = rst.read_text(encoding="utf-8")
        dst = rst_to_md(src, rst_path=rst, docs_root=root)

        if not args.dry_run:
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(dst, encoding="utf-8")

        created += 1

    print(f"RST files found: {len(rst_files)}")
    if args.dry_run:
        print(f"Would generate: {created} markdown files")
    else:
        print(f"Generated: {created} markdown files")
    if skipped:
        print(f"Skipped existing: {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
