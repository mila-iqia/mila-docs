"""Tests that launch the examples as jobs on the Mila cluster and check that they work correctly."""

from __future__ import annotations

import logging
import os
import re
import runpy
import shlex
import subprocess
import time
from logging import getLogger
from pathlib import Path

import pytest
import rich.console
import rich.logging
import rich.traceback
from pytest_regressions.file_regression import FileRegressionFixture

from .testutils import (
    EXAMPLES_DIR,
    WORK_DIR,
    TEST_JOB_NAME,
    SlurmJob,
    filter_job_output_before_regression_check,
    run_example,
)

logger = getLogger(__name__)
SCRATCH = Path(os.environ["SCRATCH"])


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging (using a recipe from @JesseFarebro)"""
    # Get the current logging level:
    # NOTE: Pytest already sets the logging level with --log-level, so we don't do it here.
    level = logger.getEffectiveLevel()
    console = rich.console.Console()

    _TRACEBACKS_EXCLUDES = [
        runpy,
        "absl",
        "click",
        "tyro",
        "simple_parsing",
        "fiddle",
    ]

    rich.traceback.install(
        console=console, suppress=_TRACEBACKS_EXCLUDES, show_locals=False
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                tracebacks_suppress=_TRACEBACKS_EXCLUDES,
            )
        ],
    )


def _test_id(arg: Path | bool | dict) -> str:
    if isinstance(arg, Path):
        path = arg
        return str(path.relative_to(EXAMPLES_DIR))
    if isinstance(arg, bool):
        return str(arg)
    assert isinstance(arg, list)
    return "_".join(arg)


@pytest.mark.parametrize(
    ("example_dir", "make_reproducible", "test_only", "sbatch_options"),
    [
        (EXAMPLES_DIR / "frameworks" / "pytorch_setup", False, False, []),
        (EXAMPLES_DIR / "distributed" / "single_gpu", True, False, []),
        (EXAMPLES_DIR / "distributed" / "multi_gpu", True, False, []),
        (EXAMPLES_DIR / "distributed" / "multi_node", True, False, []),
        (EXAMPLES_DIR / "good_practices" / "launch_many_jobs", True, False, []),
        (EXAMPLES_DIR / "good_practices" / "many_tasks_per_gpu", True, False, []),
        (EXAMPLES_DIR / "good_practices" / "slurm_job_arrays", True, False, ["--array", "1-5"]),
        (EXAMPLES_DIR / "good_practices" / "wandb_setup", True, False, []),
        (EXAMPLES_DIR / "advanced" / "imagenet", True, True, []),
        # Jax examples
        (EXAMPLES_DIR / "frameworks" / "jax_setup", False, False, []),
        (EXAMPLES_DIR / "frameworks" / "jax", False, False, []),
    ],
    ids=_test_id,
)
def test_example(
    example_dir: Path,
    make_reproducible: bool,
    sbatch_options: list[str],
    test_only: bool,
    file_regression: FileRegressionFixture,
):
    """Launches a pytorch-based example as a slurm job and checks that the output is as expected.

    Some of the examples are modified so their outputs are reproducible.
    """

    subprocess.run(["sbatch", "--test-only", example_dir / "job.sh"], check=True)

    if test_only:
        return

    filtered_job_outputs = run_example(
        example_dir=example_dir,
        sbatch_options=sbatch_options,
        examples_dir=EXAMPLES_DIR,
        make_reproducible=make_reproducible,
    )
    if len(filtered_job_outputs) == 1:
        # Only one task.
        file_regression.check(filtered_job_outputs[0])
    else:
        file_regression.check(
            "\n".join(
                [
                    f"Task {i} output:\n" + task_i_output
                    for i, task_i_output in enumerate(filtered_job_outputs)
                ]
            )
        )


def test_checkpointing_example(file_regression: FileRegressionFixture):
    """Tests the checkpointing example.

    This test is quite nice. Here's what it does:
    - Launch the job, let it run till completion.
    - Launch the job again, and then do `scontrol requeue <job_id>` to force it
      to be requeued once it has created a checkpoint (reached Epoch 1)
    - Check that the exact same result is reached whether it is requeued or not.
    """
    def exit_on_signal(python_script: Path):
        """Modifies the Python script to exit once it receives a signal."""
        python_script_content = python_script.read_text()
        python_script_content = python_script_content.replace(
            'logger.error(f"Job received a {signal_enum.name} signal!")',
            'logger.error(f"Job received a {signal_enum.name} signal!") ; import time ; time.sleep(1) ; exit(1)'
        )
        python_script.write_text(python_script_content)
    
    def sleep_on_epoch(python_script: Path):
        """Modifies the Python script to sleep for 5 seconds on each epoch."""
        python_script_content = python_script.read_text()
        python_script_content = python_script_content.replace(
            'logger.debug(f"Starting epoch {epoch}/{epochs}")',
            'import time ; time.sleep(5) ; logger.debug(f"Starting epoch {epoch}/{epochs}")'
        )
        python_script.write_text(python_script_content)
    
    def skip_lines_until(content: str, pattern: str) -> str:
        """Skips lines until the first line that matches the pattern is found."""
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                return "\n".join(lines[i:])
        return content

    example_dir = EXAMPLES_DIR / "good_practices" / "checkpointing"
    test_example_dir = WORK_DIR / "_".join(example_dir.relative_to(EXAMPLES_DIR).parts)

    uninterrupted_job_outputs = run_example(
        example_dir=example_dir,
        test_example_dir=test_example_dir,
        make_reproducible=True,
    )
    assert len(uninterrupted_job_outputs) == 1
    uninterrupted_job_output = uninterrupted_job_outputs[0]
    file_regression.check(uninterrupted_job_output)

    # NOTE: Reusing the exact same job.sh and main.py scripts as were used above:
    job_script = test_example_dir / "job.sh"
    python_script = test_example_dir / "main.py"
    # Make the execution exit once it receives a signal.
    exit_on_signal(python_script)
    # Make the training take longer per epoch.
    sleep_on_epoch(python_script)

    job = SlurmJob(
        [job_script],
        job_name="_".join([TEST_JOB_NAME, job_script.parent.name]),
        job_dir=job_script.parent,
    )
    job.submit()

    interval_seconds = 1

    while job.get_state() in ["UNKNOWN", "PENDING"]:
        logger.debug(f"Waiting for job {job.job_id} to start running. ({job.get_state()=!r})")
        time.sleep(interval_seconds)
    assert job.get_state() == "RUNNING"

    for l in job.follow_output(0):
        if "Epoch 0" in l:
            break
        else:
            logger.debug(
                f"Waiting for job {job.job_id} to reach the second epoch of training. {job.output_files()=}"
            )

    requeue_command = f"scontrol requeue {job.job_id}"
    logger.info(f"Requeueing the job using {requeue_command=!r}")
    subprocess.run(shlex.split(requeue_command), check=True)

    # TODO: double-check that there aren't other intermediate states I might miss because of the low
    # time-resolution.

    while job.get_state() == "RUNNING":
        logger.debug(f"Waiting for job {job.job_id} to get requeued. ({job.get_state()=!r})")
        time.sleep(interval_seconds)

    # assert job.state == "REQUEUED"
    logger.debug(f"Job {job.job_id} is being requeued.")
    while job.get_state() == "REQUEUED":
        logger.debug(
            f"Waiting for job {job.job_id} to become pending. ({job.get_state()=!r})"
        )
        time.sleep(interval_seconds)

    logger.debug(f"Job {job.job_id} is now pending.")
    while job.get_state() == "PENDING":
        logger.debug(
            f"Waiting for job {job.job_id} to start running again. ({job.get_state()=!r})"
        )
        time.sleep(interval_seconds)

    assert job.get_state() in ["RUNNING", "COMPLETED"]
    logger.info(f"Job {job.job_id} is now running again after having been requeued.")
    # Wait for the job to finish (again):
    job.wait()
    requeued_job_output = job.read_outputs()[0]
    # Filter out lines that may change between executions:
    filtered_requeued_job_output = filter_job_output_before_regression_check(
        requeued_job_output
    )
    # TODO: Here it *might* be a bad idea for this requeued output to be checked using the
    # file_regression fixture, because it could happen that we resume from a different epoch,
    # depending on a few things:
    # - how fast the output file can actually show us that the job has reached the second epoch
    # - how long the job takes to actually stop and get requeued
    # - how fast an epoch takes to run (if this were to become << the interval at which we check the
    #   output, then we might miss the second epoch)
    file_regression.check(filtered_requeued_job_output, extension="_requeued.txt")

    # TODO: Compare the output of the requeued job to the output of the non-requeued job in a way
    # that isn't too too hard-coded for that specific example.
    # For example, we could extract the accuracies at each epoch and check that they line up.
    uninterrupted_values = get_val_loss_and_accuracy_at_each_epoch(
        uninterrupted_job_output
    )
    interrupted_values = get_val_loss_and_accuracy_at_each_epoch(
        skip_lines_until(filtered_requeued_job_output, r".*\sResuming training at epoch")
    )

    resumed_epoch = min(interrupted_values.keys())
    final_epoch = max(interrupted_values.keys())
    assert set(uninterrupted_values.keys()) > set(interrupted_values.keys())
    for epoch in range(resumed_epoch, final_epoch + 1):
        # Compare the values at each epoch, they should match:
        assert uninterrupted_values[epoch] == interrupted_values[epoch]


def get_val_loss_and_accuracy_at_each_epoch(
    filtered_job_output: str,
) -> dict[int, tuple[float, float]]:
    # [(date) (time)] INFO     Epoch 3: Val loss: 37.565 accuracy: 67.58%
    # [(date) (time)] INFO     Epoch 4: Val loss: 37.429 accuracy: 68.14%
    # [(date) (time)] INFO     Epoch 5: Val loss: 40.469 accuracy: 66.78%
    # [(date) (time)] INFO     Epoch 6: Val loss: 48.439 accuracy: 63.78%
    # [(date) (time)] INFO     Epoch 7: Val loss: 38.182 accuracy: 71.46%
    # [(date) (time)] INFO     Epoch 8: Val loss: 40.733 accuracy: 70.60%
    # [(date) (time)] INFO     Epoch 9: Val loss: 44.822 accuracy: 69.96%
    val_losses_and_accuracies: dict[int, tuple[float, float]] = {}
    for line in filtered_job_output.splitlines():
        match_epoch = re.search(r"Epoch (\d+):", line)
        match_val_loss = re.search(r"Val loss: (\d+\.\d+)", line)
        match_val_accuracy = re.search(r"accuracy: (\d+\.\d+)%", line)
        if (
            match_epoch is not None
            and match_val_loss is not None
            and match_val_accuracy is not None
        ):
            epoch = int(match_epoch.group(1))
            val_loss = float(match_val_loss.group(1))
            val_accuracy = float(match_val_accuracy.group(1))
            val_losses_and_accuracies[epoch] = (val_loss, val_accuracy)
    if not val_losses_and_accuracies:
        raise RuntimeError(
            "Unable to extract the val loss and accuracy! Perhaps the regex here are wrong?"
        )
    return val_losses_and_accuracies


def test_orion_example(file_regression: FileRegressionFixture, tmp_path: Path):
    """Tests the "HPO with Orion" example.
    """
    example_dir = EXAMPLES_DIR / "good_practices" / "hpo_with_orion"

    def modify_job_script_before_running(job_script_path: Path) -> None:
        job_script_lines = job_script_path.read_text().splitlines()

        orion_config_line_index, orion_config_line = next(
            filter(
                lambda x: "--config $ORION_CONFIG" in x[1],
                enumerate(job_script_lines)
            )
        )

        example_dir = job_script_path.parent

        import yaml

        orion_config_path = example_dir / f"orion_config_{tmp_path.name}.yaml"
        with open(orion_config_path, "w+") as f:
            yaml.dump(
                {
                    "experiment": {
                        "name": "orion-example",
                        "algorithms": {
                            "tpe": {
                                "seed": 42,
                                "n_initial_points": 5
                            }
                        },
                        "max_broken": 3,
                        "max_trials": 3,
                    },
                    "storage": {
                        "database": {
                            "host": str(tmp_path / "orion.pkl"),
                            "type": "pickleddb",
                        },
                    },
                },
                f,
            )

        orion_config_line = orion_config_line.replace("--config $ORION_CONFIG", f"--config {orion_config_path}")

        job_script_lines[orion_config_line_index] = orion_config_line
        job_script_path.write_text("\n".join(job_script_lines))

    filtered_job_outputs = run_example(
        example_dir=example_dir,
        make_reproducible=True,
        modify_job_script_before_running=modify_job_script_before_running,
    )

    assert len(filtered_job_outputs) == 1
    file_regression.check(filtered_job_outputs[0])
