from __future__ import annotations

from dataclasses import dataclass
import itertools
import os
import re
import shutil
import subprocess
import time
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Literal, Sequence, overload

logger = get_logger(__name__)

TEST_JOB_NAME = "_".join(filter(None, ["minimal_ex_tests", os.environ.get("SLURM_JOB_ID")]))
ROOT_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT_DIR / "docs" / "examples"
TESTS_DIR = Path(__file__).parent
WORK_DIR = (Path(os.environ["SCRATCH"]) / "tmp" if os.environ.get("SCRATCH") else TESTS_DIR) / ".workdir"


@dataclass
class SlurmJob:
    command: list[str]
    job_name: str
    job_dir: Path
    job_id: int | None = None
    output_pattern: str = "slurm-%A-%t-%a.out"
    error_pattern: str = "slurm-%A.err"
    extra_options: list[str] = tuple()
    uv_project_env: Path | None = Path("/tmp/venv")

    def submit(self) -> None:
        self.job_id = int(
            subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--chdir", self.job_dir,
                    "--job-name", self.job_name,
                    "--output", self.output_pattern,
                    "--error", self.error_pattern,
                    # "--partition", "main",
                    *self.extra_options,
                    *self.command,
                ],
                stdout=subprocess.PIPE,
                check=True,
                # Unset SLURM environment variables to avoid conflicts with the
                # minimal examples sbatch settings. This avoids issues like:
                # srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive.
                env={
                    k:v
                    for k,v in os.environ.items()
                    if not k.startswith("SLURM_")
                }
            ).stdout.decode().strip()
        )
    
    def patch(self) -> None:
        if Path(self.command[0]).suffix == ".sh":
            job_script = Path(self.command[0])
            assert job_script.exists() and job_script.is_file()

            content = job_script.read_text()
            content = content.replace(
                "srun", f"srun --output '{self.output_pattern}'"
            )
            content_lines = content.splitlines()
            first_non_comment_line = next(
                filter(
                    lambda i_line: i_line[1].strip() and not i_line[1].strip().startswith("#"),
                    enumerate(content_lines)
                )
            )[0]
            # Using --export=UV_PROJECT_ENVIRONMENT=<path> in srun or sbatch
            # is failing with:
            # [2025-11-26T23:41:09.626] error: execve(): bash: No such file or directory
            # Exporting UV_PROJECT_ENVIRONMENT=<path> does works.
            if self.uv_project_env is not None:
                uv_project_env = (
                    self.uv_project_env / self.job_name
                    if self.uv_project_env.is_absolute()
                    else self.job_dir /self.uv_project_env
                )
                content_lines.insert(first_non_comment_line, f"export UV_PROJECT_ENVIRONMENT={uv_project_env}")
            content_lines.insert(first_non_comment_line, "export RUST_LOG=uv=error")

            job_script.write_text("\n".join(content_lines))

    def wait(self) -> None:
        while not self.is_done():
            time.sleep(5)
    
    def is_done(self) -> bool:
        return not subprocess.run(
            ["squeue", "--job", str(self.job_id), "--noheader"],
            capture_output=True
        ).stdout.strip()

    def is_started(self) -> bool:
        return self.get_state() != "REQUEUED" and subprocess.run(
            ["sacct", "--job", f"{self.job_id}", "--noheader", "--parsable2", "--format", "Start"],
            capture_output=True,
            check=True
        ).stdout.decode().splitlines()[0].strip() != "Unknown"

    def get_state(self) -> str:
        return subprocess.run(
            ["sacct", "--job", f"{self.job_id}", "--noheader", "--parsable2", "--format", "State"],
            capture_output=True,
            check=True
        ).stdout.decode().splitlines()[0].strip()

    def output_files(self, task_id: int | None = None):
        return sorted(
            _f
            for _f in self.job_dir.glob(
                self.output_pattern.replace("%A", str(self.job_id))
                .replace("%a", "*")
                .replace("%J", f"{self.job_id}*")
                .replace("%j", str(self.job_id))
                .replace("%t", (str(task_id) if task_id is not None else "*"))
            )
            if not any(pattern in _f.name for pattern in ("%A", "%a", "%J", "%j", "%t", "%s"))
        )

    def read_outputs(self) -> str:
        # Check if the job was successful
        state = self.get_state()
        if state != "COMPLETED":
            logger.error(
                f"Job {self.job_id} is not ready to read outputs as it's state is not COMPLETED: {state}"
            )

        return [_f.read_text() for _f in self.output_files()]

    def follow_output(self, task_id: int | None = None, timeout: int | None = None):
        start_time = time.time()
        def wait(seconds: int):
            time.sleep(seconds)
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Following output of job {self.job_id} timed out after {timeout} seconds")

        while not self.is_done() and not self.output_files(task_id):
            wait(5)
        
        output_file = self.output_files(task_id)[0]

        if not output_file.exists():
            return

        with output_file.open("rt") as _f:
            while not self.is_done():
                line = _f.readline()
                if not line:
                    wait(5)
                else:
                    yield line
            yield from _f.readlines()


@overload
def submit_example(
    job_script: Path,
    sbatch_options: list[str] = tuple(),
    wait_for_results: Literal[True] = True,
) -> list[str]:
    ...


@overload
def submit_example(
    job_script: Path,
    sbatch_options: list[str] = tuple(),
    wait_for_results: Literal[False] = False,
) -> SlurmJob:
    ...


@overload
def submit_example(
    job_script: Path,
    sbatch_options: list[str] = tuple(),
    wait_for_results: bool = True,
) -> list[str] | SlurmJob:
    ...


def submit_example(
    job_script: Path,
    sbatch_options: list[str] = tuple(),
    wait_for_results: bool = True,
) -> list[str] | SlurmJob:
    """Submits the `job.sh` script of an example as a slurm job and returns the output.

    NOTE: The backslashes in the docstring here are there so that the IDE (VsCode) shows the full
    text when hovering over an argument.

    Parameters:
        job_script: The path to the `job.sh` script of the example to run.
        sbatch_options: SBATCH options to add.
        wait_for_results: Whether to wait for the job to finish and return results, or just submit \
            the job and return it.
    """
    assert job_script.exists() and job_script.is_file() and job_script.suffix == ".sh"
    job_dir = job_script.parent

    job = SlurmJob(
        [job_script],
        job_name="_".join([TEST_JOB_NAME, job_dir.name]),
        job_dir=job_dir,
        extra_options=sbatch_options
    )

    job.patch()
    job.submit()
    if wait_for_results:
        job.wait()
        return job.read_outputs()
    return job


def run_example(
    example_dir: str | Path,
    sbatch_options: list[str] = [],
    test_example_dir: Path | None = None,
    examples_dir: Path = EXAMPLES_DIR,
    make_reproducible: bool = True,
    work_dir: Path = WORK_DIR,
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
    if test_example_dir is None:
        test_example_dir = work_dir / "_".join(example_dir.relative_to(examples_dir).parts)
    copy_example_files_to_test_dir(example_dir, test_example_dir)

    if make_reproducible:
        logger.info(
            f"Making a variant of the main.py and job.sh files from {example_dir} to make them "
            f"~100% reproducible."
        )
        make_reproducible_version_of_example(test_example_dir)

    job_script = test_example_dir / "job.sh"
    if modify_job_script_before_running:
        modify_job_script_before_running(job_script)
    
    job_outputs = submit_example(
        job_script,
        sbatch_options=sbatch_options,
        wait_for_results=True,
    )
    # Filter out lines that may change between executions:
    return [
        filter_job_output_before_regression_check(
            job_output,
            test_example_dir=test_example_dir,
            regex_substitutions={
                r"Val loss: \d+\.\d+ accuracy: \d+\.\d+%": "Val loss: X.XX accuracy: X.XX%",
            } if not make_reproducible else None
        )
        for job_output in job_outputs
    ]


def copy_example_files_to_test_dir(
    example_dir: Path, test_example_dir: Path, include_patterns: Sequence[str] = ("*.py", "*.sh", "*.toml")
) -> None:
    test_example_dir.mkdir(exist_ok=True, parents=True)
    for file in itertools.chain(*[example_dir.glob(pattern) for pattern in include_patterns]):
        dest = test_example_dir / file.name
        if dest.exists():
            dest.unlink()
        shutil.copy2(file, dest)


def make_reproducible_version_of_example(test_example_dir: Path) -> None:
    """Create a reproducible version of the examples by inserting some code blocks in the files.

    This modifies the job.sh and main.py scripts in the test example directory.
    """
    assert test_example_dir.is_dir()

    # Modify the Python script in-place to make it ~100% reproducible:
    python_script = test_example_dir / "main.py"
    job_script = test_example_dir / "job.sh"

    python_script_content = python_script.read_text()
    job_script_content = job_script.read_text()

    def uncomment_block(content: str, block_flag: str) -> str:
        begin_line = f"## === {block_flag} ==="
        end_line = f"## === {block_flag} (END) ==="
        begin = next(iter(content.split(begin_line, 1)[1:]), None)
        block = begin and begin.split(end_line, 1)[0].strip() or ""
        return content.replace(
            block,
            "\n".join((l[2:] for l in block.splitlines() if l.startswith("# ")))
        )

    python_script_content = uncomment_block(python_script_content, "Reproducibility")
    job_script_content = uncomment_block(job_script_content, "Reproducibility")

    python_script.write_text(python_script_content)
    job_script.write_text(job_script_content)


def filter_job_output_before_regression_check(
    job_output: str,
    test_example_dir: Path = None,
    filter_on_patterns: str | tuple[str, ...] = (
        r".*\sINFO\s",
        r".*\sEpoch \d+:",
        r"^Done!$",
        r"^PyTorch.*:",
        r"^PyTorch Distributed available",
        r"^Jax.*:",
        r"^\s+GPU \d+:",
        r"^Backends:",
        r"^\s+Gloo:",
        r"^\s+NCCL:",
        r"^\s+MPI:",
    ),
    prefix_of_lines_to_remove: str | tuple[str, ...] = (
        "Date:",
        "Hostname:",
        "INFO:__main__:",
        "warning:"
    ),
    regex_substitutions: dict[str, str] = None,
    # line_matches: Sequence[str] = ("INFO:__main__:Epoch", "accuracy:"),
    line_matches: Sequence[str] = (),
) -> str:
    regex_substitutions: dict[str, str] = {
        "/Tmp/slurm.[0-9]+.0/": "$SLURM_TMPDIR/",
        r"\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]": "[(DATE) (TIME)]",
        r"\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(.\d{3})?\]?": "[(DATE)T(TIME)]",
        r"(JOB|STEP) \d+(\.\d+)? ON [\w]+-\w\d{3}": "JOB $SLURM_JOB_ID ON $SLURM_JOB_NODELIST",
        r"/network/scratch/[a-z]{1}/[a-z]+/": "$SCRATCH/",
        r"/checkpointing_example/[\d]*/checkpoints": (
            "/checkpointing_example/$SLURM_JOB_ID/checkpoints"
        ),
        r"\.py:\d+\n": ".py:LINE\n",
        r"/[^:\n]+/python3(\.\d+)?/site-packages/": "PYTHON_ENV/site-packages/",
        rf"{TEST_JOB_NAME}": "TEST_JOB_NAME",
        # UV outputs
        r"Using CPython \d+\.\d+\.\d+": "Using CPython X.XX.XX",
        r"Installed \d+ packages in \d+\.\d+s": "Installed X packages in X.XXs",
        **(regex_substitutions or {}),
    }

    if test_example_dir is not None:
        regex_substitutions[rf"{str(test_example_dir.resolve())}"] = "."

    lines = [l for l in job_output.splitlines() if not l.startswith(prefix_of_lines_to_remove)]
    line_check = lambda line: any(re.match(pattern, line) for pattern in filter_on_patterns)
    first_match_idx = next((i for i, line in enumerate(lines) if line_check(line)), 0)
    last_match_idx = next((i for i, line in enumerate(lines[::-1]) if line_check(line)), 0)
    lines = lines[first_match_idx:len(lines)-last_match_idx]
    outputs = "\n".join(lines)
    for regex, replacement in regex_substitutions.items():
        outputs = re.sub(regex, replacement, outputs)

    return "\n".join(
        line
        for line in outputs.splitlines()
        if not any(pattern in line for pattern in line_matches)
    )
