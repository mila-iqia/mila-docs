"""Tests that launch the examples as jobs on the Mila cluster and check that they work correctly."""
from __future__ import annotations

import logging
import os
import re
import runpy
import shlex
import subprocess
import time
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import pytest
import rich.console
import rich.logging
import rich.traceback
from pytest_regressions.file_regression import FileRegressionFixture

from .testutils import (
    DEFAULT_SBATCH_PARAMETER_OVERRIDES,
    EXAMPLES_DIR,
    SUBMITIT_DIR,
    TEST_JOB_NAME,
    copy_example_files_to_test_dir,
    filter_job_output_before_regression_check,
    run_example,
    run_pytorch_example,
)

logger = get_logger(__name__)
SCRATCH = Path(os.environ["SCRATCH"])


gpu_types = [
    "1g.10gb",  # MIG-ed A100 GPU
    "2g.20gb",  # MIG-ed A100 GPU
    "3g.40gb",  # MIG-ed A100 GPU
    # "a100",
    # "a100l",  # Note: needs a reservation.
    # "a6000",
    "rtx8000",
    pytest.param(
        "v100",
        marks=[
            pytest.mark.xfail(reason="Can take a while to schedule"),
            pytest.mark.timeout(120),
        ],
    ),
]


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging (using a recipe from @JesseFarebro)"""

    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    console = rich.console.Console()

    _TRACEBACKS_EXCLUDES = [
        runpy,
        "absl",
        "click",
        "tyro",
        "simple_parsing",
        "fiddle",
    ]

    rich.traceback.install(console=console, suppress=_TRACEBACKS_EXCLUDES, show_locals=False)
    logging.basicConfig(
        level=LOGLEVEL,
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


def make_conda_env_for_test(
    make_env_sh_file: Path,
    env_name_in_script: str,
    env_path: Path,
):
    job_script = make_env_sh_file
    example_dir = make_env_sh_file.parent

    # Copy all the python and .sh files from the example dir to the test example dir.
    # (NOTE: This is so we can potentially modify the contents before running them in tests.)
    test_example_dir = SUBMITIT_DIR / "_".join(example_dir.relative_to(EXAMPLES_DIR).parts)
    copy_example_files_to_test_dir(example_dir, test_example_dir)

    outputs = run_example(
        job_script=test_example_dir / job_script.name,
        conda_env_name_in_script=env_name_in_script,
        conda_env=env_path,
        sbatch_parameter_overrides=DEFAULT_SBATCH_PARAMETER_OVERRIDES,
    )
    assert len(outputs) == 1
    output = outputs[0]
    assert output.isspace(), output
    return env_path


@pytest.fixture(scope="session")
def pytorch_conda_env() -> Path:
    """A fixture that launches a job to create the PyTorch + orion conda env."""
    env_name_in_script = "pytorch"  # Name in the example
    env_name = "pytorch_test"  # Name used in the tests
    env_path = SCRATCH / "conda" / env_name
    make_env_sh_file = EXAMPLES_DIR / "frameworks" / "pytorch_setup" / "make_env.sh"
    command_to_test_that_env_is_working = (
        f"conda run --prefix {env_path} python -c 'import torch, tqdm, rich'"
    )

    try:
        subprocess.check_call(shlex.split(command_to_test_that_env_is_working))
    except subprocess.CalledProcessError:
        logger.info(f"The {env_path} env has not already been created at {env_path}.")
    else:
        logger.info(
            f"The {env_path} env has already been created with all required packages at {env_path}."
        )
        return env_path

    make_conda_env_for_test(
        env_path=env_path,
        make_env_sh_file=make_env_sh_file,
        env_name_in_script=env_name_in_script,
    )
    return env_path


@pytest.fixture(autouse=True, scope="session")
def scancel_jobs_after_tests():
    yield
    username = os.environ["USER"]
    subprocess.check_call(["scancel", "-u", username, "--name", TEST_JOB_NAME])


def _test_id(arg: Path | bool | dict) -> str:
    if isinstance(arg, Path):
        path = arg
        return str(path.relative_to(EXAMPLES_DIR))
    if isinstance(arg, bool):
        return str(arg)
    assert isinstance(arg, dict)
    return "-".join(f"{k}={v}" for k, v in arg.items())


@pytest.mark.parametrize(
    ("example_dir", "make_reproducible", "sbatch_overrides"),
    [
        pytest.param(
            EXAMPLES_DIR / "frameworks" / "pytorch_setup",
            False,
            {"gres": f"gpu:{gpu_type}:1"},
            marks=(
                [
                    pytest.mark.xfail(reason="Can take a while to schedule"),
                    pytest.mark.timeout(120),
                ]
                if gpu_type == "v100"
                else []
            ),
        )
        for gpu_type in [
            "1g.10gb",  # MIG-ed A100 GPU
            "2g.20gb",  # MIG-ed A100 GPU
            "3g.40gb",  # MIG-ed A100 GPU
            # "a100",
            # "a100l",  # Note: needs a reservation.
            # "a6000",
            "rtx8000",
            "v100",
        ]
    ]
    + [
        (EXAMPLES_DIR / "distributed" / "001_single_gpu", True, {}),
        (EXAMPLES_DIR / "distributed" / "002_multi_gpu", True, {}),
        pytest.param(
            EXAMPLES_DIR / "distributed" / "003_multi_node",
            True,
            {"partition": "long"},
            marks=[
                # pytest.mark.timeout(300),
                # pytest.mark.xfail(raises=)
            ],
        ),
    ],
    ids=_test_id,
)
def test_pytorch_example(
    example_dir: Path,
    make_reproducible: bool,
    sbatch_overrides: dict[str, Any] | None,
    pytorch_conda_env: Path,
    file_regression: FileRegressionFixture,
):
    """Launches a pytorch-based example as a slurm job and checks that the output is as expected.

    Some of the examples are modified so their outputs are reproducible.
    """

    filtered_job_outputs = run_pytorch_example(
        example_dir=example_dir,
        pytorch_conda_env_location=pytorch_conda_env,
        sbatch_parameter_overrides=sbatch_overrides,
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


@pytest.mark.timeout(10 * 60)
def test_checkpointing_example(pytorch_conda_env: Path, file_regression: FileRegressionFixture):
    """Tests the checkpointing example.

    This test is quite nice. Here's what it does:
    - Launch the job, let it run till completion.
    - Launch the job again, and then do `scontrol requeue <job_id>` to force it
      to be requeued once it has created a checkpoint (reached Epoch 1)
    - Check that the exact same result is reached whether it is requeued or not.
    """
    example_dir = EXAMPLES_DIR / "good_practices" / "checkpointing"
    test_example_dir = SUBMITIT_DIR / "_".join(example_dir.relative_to(EXAMPLES_DIR).parts)

    uninterrupted_job_outputs = run_pytorch_example(
        example_dir=example_dir,
        pytorch_conda_env_location=pytorch_conda_env,
        # Need to specify a GPU so the results are reproducible.
        sbatch_parameter_overrides={"gpus_per_task": "rtx8000:1"},
        test_example_dir=test_example_dir,
        examples_dir=EXAMPLES_DIR,
        make_reproducible=True,
    )
    assert len(uninterrupted_job_outputs) == 1
    uninterrupted_job_output = uninterrupted_job_outputs[0]
    file_regression.check(uninterrupted_job_output)

    # NOTE: Reusing the exact same job.sh and main.py scripts as were used above:
    job_script = test_example_dir / "job.sh"
    job = run_example(
        job_script,
        conda_env=pytorch_conda_env,
        conda_env_name_in_script="pytorch",
        sbatch_parameter_overrides={"gpus_per_task": "rtx8000:1"},
        wait_for_results=False,
    )
    interval_seconds = 5

    while job.state in ["UNKNOWN", "PENDING"]:
        logger.debug(f"Waiting for job {job.job_id} to start running. ({job.state=!r})")
        time.sleep(interval_seconds)
    assert job.state == "RUNNING"

    output_file = job.paths.stdout
    while not output_file.exists() or "Train epoch 1:" not in output_file.read_text():
        output_path = output_file.relative_to(Path.cwd())
        logger.debug(
            f"Waiting for job {job.job_id} to reach the second epoch of training. {output_path=}"
        )
        time.sleep(interval_seconds)

    requeue_command = f"scontrol requeue {job.job_id}"
    logger.info(f"Requeueing the job using {requeue_command=!r}")
    subprocess.check_call(shlex.split(requeue_command))

    # todo: double-check that there aren't other intermediate states I might miss because of the low
    # time-resolution.

    while job.state == "RUNNING":
        logger.debug(f"Waiting for job {job.job_id} to get requeued. ({job.state=!r})")
        time.sleep(interval_seconds)

    # assert job.state == "REQUEUED"
    logger.debug(f"Job {job.job_id} is being requeued.")
    while job.state == "REQUEUED":
        logger.debug(f"Waiting for job {job.job_id} to become pending. ({job.state=!r})")
        time.sleep(interval_seconds)

    # NOTE: The state doesn't get updated back to `RUNNING` after doing REQUEUED -> PENDING!
    # (Either that, or there's some sort of caching mechanism that would take too long to get
    # assert job.state == "PENDING"
    logger.debug(f"Job {job.job_id} is now pending.")
    # invalidated.) Therefore manually trigger a "cache" update here.
    while job.watcher.get_state(job.job_id, mode="force") == "PENDING":
        logger.debug(f"Waiting for job {job.job_id} to start running again. ({job.state=!r})")
        time.sleep(interval_seconds)

    assert job.state in ["RUNNING", "COMPLETED"]
    logger.info(f"Job {job.job_id} is now running again after having been requeued.")
    # Wait for the job to finish (again):
    requeued_job_output = job.result()
    # Filter out lines that may change between executions:
    filtered_requeued_job_output = filter_job_output_before_regression_check(requeued_job_output)
    # TODO: Here it *might* be a bad idea for this requeued output to be checked using the
    # file_regression fixture, because it could happen that we resume from a different epoch,
    # depending on a few things:
    # - how fast the output file can actually show us that the job has reached the second epoch
    # - how long the job takes to actually stop and get requeued
    # - how fast an epoch takes to run (if this were to become << the interval at which we check the
    #   output, then we might miss the second epoch)
    # ALSO: not sure if it's because we're not using `exec`, but it seems like it's taking longer
    # for the job to stop running once we ask it to requeue.
    file_regression.check(filtered_requeued_job_output, extension="_requeued.txt")

    # todo: Compare the output of the requeued job to the output of the non-requeued job in a way
    # that isn't too too hard-coded for that specific example.
    # For example, we could extract the accuracies at each epoch and check that they line up.
    uninterrupted_values = get_val_loss_and_accuracy_at_each_epoch(uninterrupted_job_output)
    interrupted_values = get_val_loss_and_accuracy_at_each_epoch(filtered_requeued_job_output)

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


@pytest.fixture(scope="session")
def pytorch_orion_conda_env() -> Path:
    """A fixture that launches a job to create the PyTorch + orion conda env."""
    env_name_in_script = "pytorch_orion"  # Name in the example
    env_name = "pytorch_orion_test"  # Name used in the tests
    env_path = SCRATCH / "conda" / env_name
    make_env_sh_file = EXAMPLES_DIR / "good_practices" / "hpo_with_orion" / "make_env.sh"
    command_to_test_that_env_is_working = (
        f"conda run --prefix {env_path} python -c 'import torch, tqdm, rich, orion'"
    )
    try:
        subprocess.check_call(shlex.split(command_to_test_that_env_is_working))
    except subprocess.CalledProcessError:
        logger.info(f"The {env_path} env has not already been created at {env_path}.")
    else:
        logger.info(
            f"The {env_path} env has already been created with all required packages at {env_path}."
        )
        return env_path

    make_conda_env_for_test(
        env_path=env_path,
        make_env_sh_file=make_env_sh_file,
        env_name_in_script=env_name_in_script,
    )
    return env_path


# TODO: Make this run faster. Times out with 10 minutes, but seems to be reaching the end though,
# which is quite strange. Perhaps we could reduce the number of trials?
@pytest.mark.timeout(20 * 60)
def test_orion_example(pytorch_orion_conda_env: Path, file_regression: FileRegressionFixture):
    """Tests the "HPO with Orion" example.

    TODO: This should probably use a different conda environment, instead of adding a
    `pip install orion` to the same pytorch env.
    """
    example_dir = EXAMPLES_DIR / "good_practices" / "hpo_with_orion"
    sbatch_overrides = None

    def modify_job_script_before_running(job_script_path: Path) -> None:
        job_script_lines = job_script_path.read_text().splitlines()
        # TODO: Make this use a database in $SLURM_TMPDIR or something, so each run is independent.

        last_line = job_script_lines[-1]
        assert "hunt" in last_line

        example_dir = job_script_path.parent
        # TODO: Create an Orion config so that we can pass the path to the database to use.
        import yaml

        orion_config_path = example_dir / "orion_config.yaml"
        with open(orion_config_path, "w+") as f:
            yaml.dump(
                {
                    "storage": {
                        "type": "legacy",
                        "database": {
                            "type": "pickleddb",
                            "host": str(example_dir / "database.pkl"),
                        },
                    },
                },
                f,
            )

        last_line = last_line.replace("--exp-max-trials 10", "--exp-max-trials 3")
        last_line = last_line.replace("hunt", f"hunt --config {orion_config_path}")

        job_script_lines[-1] = last_line
        job_script_path.write_text("\n".join(job_script_lines))

    filtered_job_outputs = run_pytorch_example(
        example_dir=example_dir,
        pytorch_conda_env_location=pytorch_orion_conda_env,
        sbatch_parameter_overrides=sbatch_overrides,
        make_reproducible=True,
        examples_dir=EXAMPLES_DIR,
        submitit_dir=SUBMITIT_DIR,
        modify_job_script_before_running=modify_job_script_before_running,
    )
    assert len(filtered_job_outputs) == 1
    file_regression.check(filtered_job_outputs[0])
