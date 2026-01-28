#!/usr/bin/env python3
"""Convert RST files to Markdown, preserving content and structure."""

import re
from pathlib import Path


def convert_rst_to_markdown(content: str) -> str:
    """Convert RST content to Markdown, preserving structure and content."""

    lines = content.split("\n")
    markdown_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Handle RST section headers with underlines
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is an underline (all same character repeated)
            if (
                next_line.strip()
                and all(c == next_line[0] for c in next_line.strip())
                and len(next_line.strip()) >= 3
                and len(next_line.strip()) >= len(line.strip())
                and line.strip()
            ):  # Ensure line is not empty
                underline_char = next_line[0]
                # Map RST underline characters to heading levels
                if underline_char == "#":
                    markdown_lines.append(f"# {line.strip()}")
                elif underline_char == "*":
                    markdown_lines.append(f"## {line.strip()}")
                elif underline_char == "=":
                    markdown_lines.append(f"### {line.strip()}")
                elif underline_char == "-":
                    markdown_lines.append(f"#### {line.strip()}")
                elif underline_char == "^":
                    markdown_lines.append(f"##### {line.strip()}")
                elif underline_char == '"':
                    markdown_lines.append(f"###### {line.strip()}")
                else:
                    markdown_lines.append(line)
                i += 2  # Skip the underline
                continue

        # Skip RST directives but preserve content
        if line.strip().startswith(".. include::"):
            i += 1
            continue

        if line.strip().startswith(".. _"):
            i += 1
            continue

        if line.strip().startswith(".. toctree::"):
            # Skip toctree directive and its indented content
            i += 1
            while i < len(lines) and (
                lines[i].startswith("   ") or lines[i].strip() == ""
            ):
                i += 1
            continue

        # Handle other directives like note, warning, code-block, prompt, etc.
        if line.strip().startswith(".. ") and "::" in line:
            directive_match = re.match(r"^\s*\.\.\s+(\w+)::", line)
            if directive_match:
                directive = directive_match.group(1)

                # Handle code-block directive
                if directive == "code-block":
                    lang_match = re.match(r"^\s*\.\.\s+code-block::\s*(\S*)", line)
                    lang = (
                        lang_match.group(1)
                        if lang_match and lang_match.group(1)
                        else ""
                    )
                    markdown_lines.append(f"```{lang}")
                    i += 1
                    # Skip blank line after directive
                    if i < len(lines) and lines[i].strip() == "":
                        i += 1
                    # Add indented content
                    while i < len(lines):
                        if lines[i].strip() == "":
                            # Preserve blank lines within code blocks
                            if i + 1 < len(lines) and lines[i + 1].startswith("   "):
                                markdown_lines.append("")
                                i += 1
                            else:
                                break
                        elif lines[i].startswith("   "):
                            # Remove 3-space indent
                            markdown_lines.append(lines[i][3:])
                            i += 1
                        else:
                            break
                    markdown_lines.append("```")
                    continue

                # Handle prompt directive
                elif directive == "prompt":
                    i += 1
                    # Skip blank line after directive
                    if i < len(lines) and lines[i].strip() == "":
                        i += 1
                    # Add content after prompt directive (usually code)
                    while i < len(lines):
                        if lines[i].strip() == "":
                            if i + 1 < len(lines) and lines[i + 1].startswith("   "):
                                markdown_lines.append("")
                                i += 1
                            else:
                                break
                        elif lines[i].startswith("   "):
                            markdown_lines.append(lines[i][3:])
                            i += 1
                        else:
                            break
                    continue

                # Handle admonition directives (note, warning, etc.)
                elif directive in [
                    "note",
                    "warning",
                    "attention",
                    "caution",
                    "danger",
                    "error",
                    "hint",
                    "important",
                    "tip",
                ]:
                    markdown_lines.append(f"> **{directive.upper()}**")
                    i += 1
                    # Skip blank line after directive
                    if i < len(lines) and lines[i].strip() == "":
                        i += 1
                    # Add indented content
                    while i < len(lines):
                        if lines[i].strip() == "":
                            # Preserve blank lines in admonitions
                            if i + 1 < len(lines) and lines[i + 1].startswith("   "):
                                markdown_lines.append(">")
                                i += 1
                            else:
                                break
                        elif lines[i].startswith("   "):
                            # Remove indent and add as quote continuation
                            markdown_lines.append("> " + lines[i][3:])
                            i += 1
                        else:
                            break
                    continue

        # Skip comment lines that are RST-specific
        if (
            line.strip().startswith(".. ")
            and "::" not in line
            and not line.strip().startswith(".. _")
        ):
            i += 1
            continue

        # Process regular content lines
        # Handle references like :ref:`text <anchor>` -> [text](#anchor)
        line = re.sub(r":ref:`([^<>]*)<([^>]*)>`", r"[\1](#\2)", line)
        line = re.sub(r":ref:`([^`]*)`", r"[\1](#\1)", line)

        # Handle external links like `text <url>`_
        line = re.sub(r"`([^<>`]+)\s*<([^>]+)>`_", r"[\1](\2)", line)

        # Handle inline code
        line = re.sub(r"``([^`]+)``", r"`\1`", line)

        # Handle bold and italic (keep as is, they're the same in RST and Markdown)
        # **text** and *text* are already valid in Markdown

        # Clean up orphaned .. _linkname: references
        if line.strip().startswith(".. _") and ":" in line:
            i += 1
            continue

        markdown_lines.append(line)
        i += 1

    content = "\n".join(markdown_lines)

    # Clean up extra blank lines (max 2 consecutive)
    content = re.sub(r"\n\n\n+", "\n\n", content)

    # Remove leading/trailing whitespace
    content = content.strip()

    return content


def convert_file(rst_path: Path, md_path: Path) -> None:
    """Convert a single RST file to Markdown."""
    with open(rst_path, "r", encoding="utf-8") as f:
        content = f.read()

    markdown_content = convert_rst_to_markdown(content)

    # Ensure directory exists
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Fixed: {rst_path.name}")


def main():
    """Main conversion function."""
    docs_dir = Path("/home/fabrice/repos/mila-docs/docs")

    # Find all MD files that need fixing
    md_files = list(docs_dir.rglob("*.md"))
    print(f"Found {len(md_files)} MD files to check/fix")

    fixed = 0
    for md_file in sorted(md_files):
        # Get corresponding RST file
        rst_file = md_file.with_suffix(".rst")

        # Skip if no RST file exists
        if not rst_file.exists():
            continue

        # Check if MD file is from one of the original ones (not converted)
        skip_files = {
            "index.md",
            "Purpose.md",
            "Acknowledgement_text.md",
            "Userguide_quick_start.md",
        }
        if md_file.name in skip_files:
            continue

        convert_file(rst_file, md_file)
        fixed += 1

    print(f"\nFix complete: {fixed} files fixed")


if __name__ == "__main__":
    main()
