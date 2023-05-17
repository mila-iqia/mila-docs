"""Generate GitHub README's from _index.rst files
GitHub doesn't support include of other files, even of the same type and
location, so this file generates a README.rst with files content embedded
"""
from glob import glob
from pathlib import Path
import shutil
import subprocess
import sys


def preprocess():
    docs_root = Path(__file__).absolute().parent.parent
    script = docs_root / "examples/generate_diffs.sh"
    try:
        _proc = subprocess.run(str(script), shell=True, check=True)
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

    for _f in glob(str(docs_root / "examples/**/_index.rst"), recursive=True):
        _f = Path(_f)
        shutil.copyfile(str(_f), str(_f.with_name("README.rst")))
        _f = _f.with_name("README.rst")
        content = _f.read_text().split("\n")
        i = 0
        end = len(content)
        while i < end:
            line = content[i]
            if line.startswith(".. literalinclude:: "):
                path = line[len(".. literalinclude:: "):].strip(" ")
                lang = ""
                for j, _l in enumerate(content[i+1:]):
                    _l = _l.strip(" ")
                    if _l.startswith(":language:"):
                        lang = _l[len(":language:"):].strip(" ")
                    elif _l.startswith(".. literalinclude:: ") or not _l:
                        break
                del content[i:i+1+j]
                insert = (
                    [f".. code:: {lang}", ""] +
                    [f"   {_l}".rstrip(" ") for _l in (docs_root / path).read_text().split("\n")]
                )
                content = content[:i] + insert + content[i+1:]
                i += len(insert)
                end = len(content)
            else:
                i += 1
        _f.write_text("\n".join(content).replace("\t", " " * 4))


if __name__ == "__main__":
    preprocess()
