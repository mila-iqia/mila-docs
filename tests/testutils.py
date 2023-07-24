""" Idea: Use `submitit` to test that the setup works for this repo on the current cluster.
"""
from __future__ import annotations

import itertools
import json
import re
import shlex
import shutil
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypeVar, overload
import submitit

logger = get_logger(__name__)

TEST_JOB_NAME = "example_tests"
ROOT_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT_DIR / "docs" / "examples"
TESTS_DIR = Path(__file__).parent
SUBMITIT_DIR = TESTS_DIR / ".submitit"

DEFAULT_SBATCH_PARAMETER_OVERRIDES = dict(
    partition="main",
    job_name=TEST_JOB_NAME,
    stderr_to_stdout=True,
)


REPRODUCIBLE_BLOCK_PYTHON = """\
### NOTE: This block is added to make the example reproducible during unit tests
import random
import numpy

seed = 123
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
###
"""

REPRODUCIBLE_BLOCK_BATCH_SCRIPT = """\
##
# Adding this line makes it possible to set `torch.use_deterministic_algorithms(True)`
export CUBLAS_WORKSPACE_CONFIG=:4096:8
##
"""


@overload
def run_example(
    job_script: Path,
    conda_env: Path,
    conda_env_name_in_script: str,
    sbatch_parameter_overrides: dict[str, Any]
    | None = None,  # Actually defaults to `default_overrides`
    wait_for_results: Literal[True] = True,
) -> list[str]:
    ...


@overload
def run_example(
    job_script: Path,
    conda_env: Path,
    conda_env_name_in_script: str,
    sbatch_parameter_overrides: dict[str, Any]
    | None = None,  # Actually defaults to `default_overrides`
    wait_for_results: Literal[False] = False,
) -> submitit.Job[str]:
    ...


@overload
def run_example(
    job_script: Path,
    conda_env: Path,
    conda_env_name_in_script: str,
    sbatch_parameter_overrides: dict[str, Any]
    | None = None,  # Actually defaults to `default_overrides`
    wait_for_results: bool = True,
) -> list[str] | submitit.Job[str]:
    ...


def run_example(
    job_script: Path,
    conda_env: Path,
    conda_env_name_in_script: str,
    sbatch_parameter_overrides: dict[str, Any]
    | None = None,  # Actually defaults to `default_overrides`
    wait_for_results: bool = True,
) -> list[str] | submitit.Job[str]:
    """Submits the `job.sh` script of an example as a slurm job and returns the output.

    NOTE: The backslashes in the docstring here are there so that the IDE (VsCode) shows the full
    text when hovering over an argument.

    Parameters:
        job_script: The path to the `job.sh` script of the example to run.
        conda_env: The path to the conda environment to use in the example.
        conda_env_name_in_script: The name of the conda environment as it appears in the `job.sh` \
            of the example. This is replaced with `conda_env` before the job is submitted.
        sbatch_parameter_overrides: SBATCH parameters to override (in python form, e.g. \
            `--ntasks-per-node` becomes "ntasks_per_node").
        wait_for_results: Whether to wait for the job to finish and return results, or just submit \
            the job and return it.
    """
    assert job_script.exists() and job_script.is_file() and job_script.suffix == ".sh"
    sbatch_parameter_overrides = sbatch_parameter_overrides or DEFAULT_SBATCH_PARAMETER_OVERRIDES
    example_dir = job_script.parent
    # Adds the --chdir parameter as a SBATCH flag, so the paths work and the outputs are produced in
    # the right folder.
    sbatch_parameter_overrides.setdefault("additional_parameters", {})["chdir"] = str(example_dir)

    job_script_content = job_script.read_text()
    job_script_content = change_conda_env_used_in_job_script(
        job_script_content,
        conda_env_path=conda_env,
        conda_env_name_in_script=conda_env_name_in_script,
    )
    example_lines_after_sbatch = [
        stripped_line
        for line in job_script_content.splitlines(keepends=False)
        if (stripped_line := line.strip()) and not stripped_line.startswith("#SBATCH")
    ]
    last_non_empty_line_index = -1
    job_setup = example_lines_after_sbatch[:last_non_empty_line_index]
    job_command_in_example = example_lines_after_sbatch[last_non_empty_line_index]

    # NOTE: Could be nice to use the new match-case statement for this, but it requires python=3.10
    # match job_command.split():
    #     case "python main.py":
    srun_args: list[str] = sbatch_parameter_overrides.get("srun_args", [])
    _old_srun_args = srun_args.copy()
    # NOTE: If there's an `srun` in the job command, this is going to cause an issue, because
    # submitit will create a last line that goes
    # `srun (...) submitit.load_and_run_ish "srun job.sh"` and the job will hang!
    # Therefore, we tweak the last line of the example into something that will work with submitit.
    submitit_job_command, srun_args = _get_submitit_job_command_and_srun_args(
        job_command_in_example, srun_args=srun_args
    )
    sbatch_parameter_overrides["srun_args"] = srun_args
    if submitit_job_command != job_command_in_example:
        logger.debug(f"{job_command_in_example=!r}")
        logger.debug(f"srun args before: {_old_srun_args!r}")
        logger.debug(f"{submitit_job_command=!r}")
        logger.debug(f"srun args after: {srun_args!r}")

    logger.info(f"Command that will be run by submitit: {submitit_job_command!r}")
    logger.info(f"Additional args to be passed to `srun`: {srun_args!r}")

    job_setup = (
        ["set -e"]  # Make the job crash if one of the command fails.
        + job_setup
        + ([f"# NOTE: Command that will be run by submitit: {submitit_job_command!r}"])
    )

    executor = submitit.SlurmExecutor(folder=example_dir)
    job_script_params = get_params_from_job_script(job_script)
    executor.update_parameters(
        setup=job_setup,
        **_recursive_dict_union(job_script_params, sbatch_parameter_overrides),
    )
    logger.debug(f"Using the following sbatch params: {json.dumps(executor.parameters, indent=4)}")

    assert "srun" not in submitit_job_command
    function = submitit.helpers.CommandFunction(
        shlex.split(submitit_job_command),
        cwd=example_dir,
    )
    job = executor.submit(function)
    if wait_for_results:
        job_outputs = job.results()
        return job_outputs
    return job


def run_pytorch_example(
    example_dir: str | Path,
    pytorch_conda_env_location: Path,
    sbatch_parameter_overrides: dict[str, Any] | None = None,
    test_example_dir: Path | None = None,
    examples_dir: Path = EXAMPLES_DIR,
    make_reproducible: bool = True,
    submitit_dir: Path = SUBMITIT_DIR,
    modify_job_script_before_running: Callable[[Path], None] | None = None,
) -> list[str]:
    """Runs a pytorch-base example with a main.py and job.sh file.

    Compared with `run_example`, this also:
    - Copies the files into a `test_example_dir` directory so they can be modified before being run
    - Optionally makes it reproducible by adding a block of code to the main.py and job.sh files
    - Filters out the job output to remove lines that may change between executions
    """
    example_dir = Path(example_dir)
    assert example_dir.is_dir()
    assert (example_dir / "job.sh").is_file()
    assert (example_dir / "main.py").is_file()
    assert example_dir.is_relative_to(examples_dir)
    assert pytorch_conda_env_location.is_dir()
    if test_example_dir is None:
        test_example_dir = submitit_dir / "_".join(example_dir.relative_to(examples_dir).parts)
    copy_example_files_to_test_dir(example_dir, test_example_dir)

    if make_reproducible:
        logger.info(
            f"Making a variant of the main.py and job.sh files from {example_dir} to make them "
            f"~100% reproducible."
        )
        make_reproducible_version_of_example(example_dir, test_example_dir)

    job_script = test_example_dir / "job.sh"
    if modify_job_script_before_running:
        modify_job_script_before_running(job_script)

    job_outputs = run_example(
        job_script,
        conda_env=pytorch_conda_env_location,
        sbatch_parameter_overrides=sbatch_parameter_overrides or {},
        conda_env_name_in_script="pytorch",
        wait_for_results=True,
    )
    # Filter out lines that may change between executions:
    return [filter_job_output_before_regression_check(job_output) for job_output in job_outputs]


def copy_example_files_to_test_dir(
    example_dir: Path, test_example_dir: Path, include_patterns: Sequence[str] = ("*.py", "*.sh")
) -> None:
    test_example_dir.mkdir(exist_ok=True, parents=True)
    for file in itertools.chain(*[example_dir.glob(pattern) for pattern in include_patterns]):
        dest = test_example_dir / file.name
        if dest.exists():
            dest.unlink()
        shutil.copyfile(file, dest)


def make_logging_use_wider_console(python_script_content: str) -> str:
    """Make the example use a wider console for logging.

    This is done so we can more easily match or substitute some of the contents of the job outputs
    before they are checked using the file regression fixture.
    """
    old = "rich.logging.RichHandler(markup=True)"
    new = "rich.logging.RichHandler(markup=True, console=rich.console.Console(width=255))"
    assert old in python_script_content
    return python_script_content.replace(old, new)


def _get_submitit_job_command_and_srun_args(
    job_command_in_example: str, srun_args: list[str]
) -> tuple[str, list[str]]:
    """Adapts the last line of the job script so it can be run using a CommandFunction of submitit.

    TODO: This needs to be customized for each example, unfortunately.
    """
    srun_args = srun_args.copy()

    if job_command_in_example == "python main.py":
        return job_command_in_example, srun_args

    if job_command_in_example == "srun python main.py":
        # submitit already does `srun (...) run_this_command_ish "python main.py"` as the last line,
        # so we just remove the srun prefix here.
        return "python main.py", srun_args

    if job_command_in_example == "exec python main.py":
        # BUG: Getting a FileNotFoundError("exec") here if we leave the `exec python main.py` in!
        return "python main.py", srun_args

    if "srun" in job_command_in_example:
        # TODO: We need to do something different if we have an `srun` in the last line!
        # Make the last line (the job command) just python main.py (...) and move all the srun args
        # into the `srun_args` list of submitit.
        # TODO: Need to remove the `srun` and move the srun params into the srun_args, so that
        # submitit can do it with a CommandFunction that uses the right conda env!
        # job_command = "python main.py"
        # raise NotImplementedError(job_command)
        srun_part, python, python_args = job_command_in_example.partition("python")
        srun_args_str = srun_part.removeprefix("srun")
        srun_args.append(srun_args_str)
        job_command = python + python_args
        return job_command, srun_args

    warnings.warn(
        RuntimeWarning(
            f"Don't yet know how to adapt the {job_command_in_example=!r} to work "
            f"with submitit. Will try to run it as-is."
        )
    )
    return job_command_in_example, srun_args


def change_conda_env_used_in_job_script(
    job_script_content: str, conda_env_path: Path, conda_env_name_in_script: str
) -> str:
    """Modify some lines of the source job script before it is run in the unit test."""
    return (
        (job_script_content)
        .replace(f"conda activate {conda_env_name_in_script}", f"conda activate {conda_env_path}")
        .replace(f"-n {conda_env_name_in_script}", f"--prefix {conda_env_path}")
        .replace(f"--name {conda_env_name_in_script}", f"--prefix {conda_env_path}")
        .replace(f"-p {conda_env_name_in_script}", f"--prefix {conda_env_path}")
        .replace(f"--prefix {conda_env_name_in_script}", f"--prefix {conda_env_path}")
    )


def make_reproducible_version_of_example(example_dir: Path, test_example_dir: Path) -> None:
    """Create a reproducible version of the examples by inserting some code blocks in the files.

    This modifies the job.sh and main.py scripts in the test example directory.
    """
    directory_with_modified_files_for_test = test_example_dir
    assert directory_with_modified_files_for_test.is_dir()

    # Modify the Python script in-place to make it ~100% reproducible:
    python_script = example_dir / "main.py"
    job_script = example_dir / "job.sh"

    modified_job_script = directory_with_modified_files_for_test / job_script.name
    modified_python_script = directory_with_modified_files_for_test / python_script.name

    python_script_content = python_script.read_text()
    python_script_content = make_logging_use_wider_console(python_script_content)
    python_script_lines = python_script_content.splitlines(keepends=False)
    # TODO: Where do we add the block? Before the def main()? Inside main?

    insertion_index = python_script_lines.index("def main():") - 1
    python_script_lines = (
        python_script_lines[:insertion_index]
        + [""]
        + REPRODUCIBLE_BLOCK_PYTHON.splitlines()
        + [""]
        + python_script_lines[insertion_index:]
    )
    job_script_lines = job_script.read_text().splitlines(keepends=False)
    insertion_index = -2
    # Somewhere before the end of the script (assuming the last line has the main command.)
    job_script_lines = (
        job_script_lines[:insertion_index]
        + REPRODUCIBLE_BLOCK_BATCH_SCRIPT.splitlines()
        + job_script_lines[insertion_index:]
    )

    modified_python_script.write_text("\n".join(python_script_lines))
    modified_job_script.write_text("\n".join(job_script_lines))


def filter_job_output_before_regression_check(
    job_output: str,
    prefix_of_lines_to_remove: str | tuple[str, ...] = ("Date:", "Hostname:", "INFO:__main__:"),
    regex_substitutions: dict[str, str] = {
        "/Tmp/slurm.[0-9]+.0/": "$SLURM_TMPDIR/",
        r"\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]": "[(date) (time)]",
        r"network/scratch/[a-z]{1}/[a-z]*/": "$SCRATCH/",
        r"/checkpointing_example/[\d]*/checkpoints": (
            "/checkpointing_example/$SLURM_JOB_ID/checkpoints"
        ),
    },
    # line_matches: Sequence[str] = ("INFO:__main__:Epoch", "accuracy:"),
    line_matches: Sequence[str] = (),
) -> str:
    outputs = "\n".join(
        line for line in job_output.splitlines() if not line.startswith(prefix_of_lines_to_remove)
    )
    for regex, replacement in regex_substitutions.items():
        outputs = re.sub(regex, replacement, outputs)

    return "\n".join(
        line
        for line in outputs.splitlines()
        if not any(pattern in line for pattern in line_matches)
    )


def get_params_from_job_script(job_script: Path) -> dict[str, Any]:
    lines = job_script.read_text().splitlines()
    sbatch_lines = [
        line.strip().removeprefix("#SBATCH").split("#", 1)[0].strip()
        for line in lines
        if line.strip().startswith("#SBATCH")
    ]
    params: dict[str, Any] = {}
    for sbatch_arg_string in sbatch_lines:
        value: Any
        if "=" not in sbatch_arg_string:
            flag = sbatch_arg_string
            value = True
        else:
            flag, _, value = sbatch_arg_string.partition("=")
            value = value.strip()
            if value.isnumeric():
                value = int(value)
        new_key = flag.strip().lstrip("-").replace("-", "_")
        params[new_key] = value
    for key in ["signal", "requeue"]:
        if key in params:
            params.setdefault("additional_parameters", {})[key] = params.pop(key)
    return params


K = TypeVar("K")
V = TypeVar("V")


def _recursive_dict_union(*dicts: dict[K, V]) -> dict[K, V]:
    """Recursively merge two dictionaries."""
    result: dict[K, V] = {}
    for key in set(dicts[0]).union(*dicts[1:]):
        values = [d[key] for d in dicts if key in d]
        if any(isinstance(value, dict) for value in values):
            result[key] = _recursive_dict_union(
                *[value for value in values if isinstance(value, dict)]
            )
        else:
            result[key] = values[-1]
    return result
