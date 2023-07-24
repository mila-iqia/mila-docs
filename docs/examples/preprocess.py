"""Generate GitHub README's from index.rst files
GitHub doesn't support include of other files, even of the same type and
location, so this file generates a README.rst with files content embedded
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import logging
from logging import getLogger as get_logger
import textwrap

logger = get_logger(__name__)
docs_root = Path(__file__).absolute().parent.parent
assert docs_root.name == "docs"


def preprocess():
    """Preprocessing for the minimal examples in the docs.

    1. Generates the diffs between examples and their base to be shown in the docs.
    2. Makes sure that each example has a link to the source code on GitHub.
    3. Makes a GitHub-friendly version of the example and saves it as a `README.rst` file.
    """

    generate_diffs(docs_root)

    # NOTE: We require each example .rst file to include a link to the source file on GitHub.
    # Top-level files in a section (e.g. docs/examples/distributed/index.rst) are exempt
    # from this requirement since all the examples in that section will have links.

    for example_readme_template_path in docs_root.rglob("examples/*/*/**/index.rst"):
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
    relative_readme_template_path = example_readme_template_path.relative_to(docs_root)
    relative_readme_folder = relative_readme_template_path.parent
    github_link = (
        f"https://github.com/mila-iqia/mila-docs/tree/master/docs/{relative_readme_folder}"
    )
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


def make_links_github_friendly(readme_template_path: Path, content_lines: list[str]) -> list[str]:
    relative_readme_template_path = readme_template_path.relative_to(docs_root)
    result = content_lines.copy()
    # # TODO: Change refs to Prerequisites from rst format into GitHub-friendly links.
    # NOTE: Seems pretty hard to do, because the link location is something like "Quick Start",
    # which isn't a file name. Perhaps if we could use filenames for the refs, that might work.
    for i, line in enumerate(content_lines):
        if "* :doc:`" not in line:
            continue
        spaces_at_start_of_line = (len(line) - len(line.lstrip(" "))) * " "
        reference = line.strip()[len("* :doc:`") :].split("`")[0]
        logger.debug(f"Example {relative_readme_template_path} has link to {reference}.")
        referenced_file = (
            docs_root / reference[1:]
            if reference.startswith("/examples")
            else relative_readme_template_path.parent / reference[1:]
        )
        referenced_file = referenced_file.relative_to(docs_root)
        referenced_folder = referenced_file.parent
        github_link = (
            f"https://github.com/mila-iqia/mila-docs/tree/master/docs/{referenced_folder}"
        )
        new_line = spaces_at_start_of_line + f"* `{referenced_folder} <{github_link}>`_"
        result[i] = new_line
        # line = ""
    return result


def inline_docs_for_github_viewing(example_readme_template_path: Path) -> str:
    relative_readme_template_path = example_readme_template_path.relative_to(docs_root)

    content = example_readme_template_path.read_text()
    content_lines = content.splitlines()
    content_lines = (
        textwrap.dedent(
            f"""\
            .. NOTE: This file is auto-generated from {relative_readme_template_path}
            .. This is done so this file can be easily viewed from the GitHub UI.
            .. **DO NOT EDIT**
            """
        ).splitlines()
        + [""]  # Need an empty line between this header and the content, otherwise rstcheck cries
        + content_lines
    )

    i = 0
    end = len(content_lines)
    while i < end:
        line = content_lines[i]
        if line.startswith(".. literalinclude:: "):
            path = line[len(".. literalinclude:: ") :].strip(" ")
            # TODO: This causes an issue when building the docs because the path to job.sh
            if path.startswith("examples/"):
                # all good, the path is absolute already.
                # logger.debug(f"The path to source code is already absolute: {path}")
                pass
            elif path.count("/") == 0:
                # logger.debug(f"The path to source code is relative: {path}")
                path = str(example_readme_template_path.parent / path)
            else:
                raise NotImplementedError(
                    f"Example {example_readme_template_path} uses a weird format in a "
                    f"literalinclude: {path}"
                )
            lang = ""
            j = 0
            for j, line in enumerate(content_lines[i + 1 :]):
                line = line.strip(" ")
                if line.startswith(":language:"):
                    lang = line[len(":language:") :].strip(" ")
                elif line.startswith(".. literalinclude:: ") or not line:
                    break
            del content_lines[i : i + 1 + j]

            insert = [f".. code:: {lang}", ""] + [
                # NOTE: The code block in the .rst files is indented with 3 spaces, but the
                # rendered code blocks in the README.rst on GitHub and in the online docs have the
                # correct indented for a Python script (4 spaces).
                f"   {_l}".rstrip(" ")
                for _l in (docs_root / path).read_text().split("\n")
            ]
            content_lines = content_lines[:i] + insert + content_lines[i + 1 :]
            i += len(insert)
            end = len(content_lines)
        else:
            i += 1

    # Write out the new contents (template + inlined files) to a README.rst that is viewable
    # from the GitHub page.
    new_content = "\n".join(content_lines).replace("\t", " " * 4) + "\n"
    return new_content


if __name__ == "__main__":
    handlers = []
    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(markup=True))
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    preprocess()
