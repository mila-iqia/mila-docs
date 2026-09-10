"""Generate GitHub README's from index.rst files

GitHub doesn't support include of other files, even of the same type and
location, so this file generates a README.rst with files content embedded.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import textwrap
from logging import getLogger as get_logger
from pathlib import Path

logger = get_logger(__name__)
DOCS_ROOT = Path(__file__).absolute().parent.parent
assert DOCS_ROOT.name == "docs"


def preprocess():
    """Preprocessing for the minimal examples in the docs.

    1. Generates the diffs between examples and their base to be shown in the docs.
    2. Makes sure that each example has a link to the source code on GitHub.
    3. Makes a GitHub-friendly version of the example and saves it as a `README.rst` file.
    """

    generate_diffs(DOCS_ROOT)

    # NOTE: We require each example .rst file to include a link to the source file on GitHub.
    # Top-level files in a section (e.g. docs/examples/distributed/index.rst) are exempt
    # from this requirement since all the examples in that section will have links.

    for example_readme_template_path in DOCS_ROOT.rglob("examples/*/*/**/index.rst"):
        # Make sure that all the examples contain a link to the example source code on GitHub.
        logger.debug(f"{example_readme_template_path}")
        check_github_links(example_readme_template_path)

        new_content = inline_docs_for_github_viewing(example_readme_template_path)
        new_content = make_links_github_friendly(
            example_readme_template_path, new_content.splitlines()
        )
        example_readme_path = example_readme_template_path.with_name("README.rst")
        example_readme_path.write_text("\n".join(new_content) + "\n")


def check_github_links(example_readme_template_path: Path):
    relative_readme_template_path = example_readme_template_path.relative_to(DOCS_ROOT)
    relative_readme_folder = relative_readme_template_path.parent
    github_link = f"https://github.com/mila-iqia/mila-docs/tree/master/docs/{relative_readme_folder}"
    content = example_readme_template_path.read_text()
    if github_link not in content:
        if "https://github.com/mila-iqia" in content:
            raise RuntimeError(
                f"The GitHub link in the example at {relative_readme_template_path} doesn't "
                f"point to the right location! It should point to {github_link}"
            )
        raise RuntimeError(
            f"The example {relative_readme_template_path} doesn't contain a link to the "
            f"example directory on GitHub: {github_link}"
        )


def generate_diffs(docs_root: Path):
    script = docs_root / "examples/generate_diffs.sh"
    try:
        subprocess.check_call(str(script), shell=True)
    except subprocess.CalledProcessError as err:
        exc = RuntimeError(
            "Could not build the diff files for the examples:\n"
            + str(err.output or b"", encoding="utf-8")
            + str(err.stderr or b"", encoding="utf-8")
        )
        # It's useful to have the genration of files in the docs compilation
        # process but it fails and throws an error related to
        # /src/docs/examples/generate_diffs.sh during the build the docs github
        # action. This hack allows to pass the exception should it happen in the
        # build the docs process. Checks are still made in
        # .github/workflows/tests.yml.
        if script.with_name(f"{script.name}_err_ok").exists():
            print(str(exc), file=sys.stderr)
        else:
            raise exc


def make_links_github_friendly(
    readme_template_path: Path, content_lines: list[str]
) -> list[str]:
    """Generates github links from :doc: refs in the example rst."""
    relative_readme_template_path = readme_template_path.relative_to(DOCS_ROOT)
    result = content_lines.copy()
    os.chdir(DOCS_ROOT)
    for i, line in enumerate(content_lines):
        if "* :doc:`" not in line:
            continue
        spaces_at_start_of_line = (len(line) - len(line.lstrip(" "))) * " "
        reference = line.strip()[len("* :doc:`") :].split("`")[0]
        logger.debug(
            f"Example {relative_readme_template_path} has link to {reference}."
        )
        referenced_file = (
            DOCS_ROOT / reference[1:]
            if reference.startswith("/examples")
            else relative_readme_template_path.parent / reference[1:]
        )
        referenced_file = referenced_file.relative_to(DOCS_ROOT)
        if not (f := referenced_file.with_suffix(".rst")).exists():
            raise RuntimeError(
                f"The example at {relative_readme_template_path} has a link to {f} "
                f"which doesn't exist! (line = {line!r})"
            )
        referenced_folder = referenced_file.parent
        github_link = f"https://github.com/mila-iqia/mila-docs/tree/master/docs/{referenced_folder}"
        new_line = spaces_at_start_of_line + f"* `{referenced_folder} <{github_link}>`_"
        result[i] = new_line
        # line = ""
    return result


def inline_docs_for_github_viewing(example_readme_template_path: Path) -> str:
    """Replaces a literalinclude block with the contents of the file it points to.

    Replace this:
    ```
    .. literalinclude:: job.sh
       :language: bash
    ```
    with this:
    .. code-block:: bash
       (...) # Contents of job.sh

    """
    relative_readme_template_path = example_readme_template_path.relative_to(DOCS_ROOT)

    content = example_readme_template_path.read_text()
    content_lines = content.splitlines()

    def _get_path_to_file(filename: str) -> Path:
        if filename.startswith("examples/"):
            # all good, the path is absolute already.
            return DOCS_ROOT / filename
        elif "/" not in filename:
            # Path is like `job.sh.diff`, which is good.
            return example_readme_template_path.parent / filename
        else:
            raise NotImplementedError(
                f"Example {example_readme_template_path} uses a weird format in a "
                f"literalinclude: {filename}"
            )

    new_content_lines: list[str] = (
        textwrap.dedent(
            f"""\
            .. NOTE: This file is auto-generated from {relative_readme_template_path}
            .. This is done so this file can be easily viewed from the GitHub UI.
            .. **DO NOT EDIT**
            """
        ).splitlines()
        + [
            ""
        ]  # Need an empty line between this header and the content, otherwise rstcheck cries
    )

    # Matches a group of non-whitespace characters after the literalinclude directive, maybe
    # followed by some whitespace.
    literalinclude_pattern = r".. literalinclude::\s*(?P<file_path>[^\s]+)\s*"

    # Matches some leading whitespace since it needs to be idented, (note: the number of spaces
    # varies, usually 3 or 4), followed by a word and potentially some whitespace at the end.
    language_pattern = r"\s+:language:\s*(?P<language>[^\s]+)\s*"

    line_index = 0
    # Using a while loop here because we want to skip two lines whenever we find a match.
    while line_index < len(content_lines):
        line = content_lines[line_index]
        # Using fullmatch with the spaces specified in the regex so that we don't match comments.
        # There is also no need to call .strip() or the result this way.
        include_block_match = re.fullmatch(literalinclude_pattern, line)
        if not include_block_match:
            # This is just a normal text line in the doc. Add it and move to the next.
            new_content_lines.append(line)
            line_index += 1
            continue

        # Can't have a literalinclude on the last line
        assert line_index + 1 < len(content_lines)
        language_match = re.fullmatch(language_pattern, content_lines[line_index + 1])
        if not language_match:
            # NOTE: Add 1 to line index so paths in logs are clickable for debugging purposes.
            raise RuntimeError(
                f"Found a literalinclude block at "
                f"{example_readme_template_path}:{line_index + 1} but it's missing a "
                f":language: directive on the following line."
            )

        if (
            line_index + 2 < len(content_lines)
            and content_lines[line_index + 2].strip() != ""
        ):
            raise RuntimeError(
                f"Expected the literalinclude block at "
                f"{example_readme_template_path}:{line_index + 1} to only have two lines, followed "
                f"by an empty line."
            )

        file_name = include_block_match.group("file_path")
        file_path = _get_path_to_file(file_name)
        language = language_match.group("language")
        logger.debug(
            f"Found a literalinclude block at {example_readme_template_path}:{line_index + 1}"
        )
        if not file_path.exists():
            raise RuntimeError(
                f"The example at {example_readme_template_path} has a literalinclude of "
                f"{file_name!r} which can't be found."
            )
        # Create the inline code block.
        # NOTE: The code block in the .rst files is indented with 3 spaces, but the
        # rendered code blocks in the README.rst on GitHub and in the online docs is indented as
        # usual for python scripts (4 spaces).
        inlined_block_lines = [f".. code:: {language}", ""]
        inlined_block_lines.extend(
            textwrap.indent(file_path.read_text(), " " * 3).splitlines()
        )
        new_content_lines.extend(inlined_block_lines)
        line_index += 2  # go to the line after the block

    # Remove any trailing whitespace, if any:
    new_content_lines = [line.rstrip() for line in new_content_lines]
    # Replace tabs with 4 spaces (otherwise rstcheck complains):
    new_content = "\n".join(new_content_lines).replace("\t", " " * 4)
    return new_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    handlers = []
    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(markup=True))
    except ImportError:
        pass

    level = logging.DEBUG if args.verbose > 0 else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    preprocess()
