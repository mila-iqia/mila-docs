import os
import subprocess
import pytest
from xdist import is_xdist_worker

from tests.testutils import TEST_JOB_NAME, logger


@pytest.hookimpl()
def pytest_sessionfinish(session):
    """Cleanup hook that runs after all tests complete, only in the master process.
    
    This ensures that when using pytest-xdist, the cleanup only happens once
    after all parallel workers have finished, not in each worker process.
    
    The pytest_sessionfinish hook is called in the controller (master) process
    after all worker processes have completed their tests.
    """
    # Only run cleanup in the master process (not in worker processes)
    if is_xdist_worker(session):
        # We're in a worker process, skip cleanup
        return

    username = os.environ["USER"]
    for job_name in subprocess.run(
        ["squeue", "-u", username, "--all", "--Format", "NAME:120"],
        stdout=subprocess.PIPE, check=True
    ).stdout.decode().splitlines()[1:]:
        if job_name.strip().startswith(f"{TEST_JOB_NAME}_"):
            subprocess.run(
                ["scancel", "-v", "-u", username, "--name", job_name.strip()],
                check=True,
            )
            logger.info(f"Cancelled job {job_name.strip()}.")

        else:
            logger.debug(
                f"Skipping job {job_name.strip()} because it doesn't start with {TEST_JOB_NAME}."
            )
