import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def pytorch_conda_env(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that launches a job that creates the "pytorch" conda environment used in examples.

    Yields the path (str) to the conda environment, to be used with `conda activate <path>`

    This makes the examples quicker to run, prevents a bunch of internet traffic and storage, etc.
    """
    conda_dir = tmp_path_factory.mktemp("conda_envs")
    conda_env_prefix = conda_dir / "pytorch"

    # TODO: Modify the file so we create the conda environment in $SLURM_TMPDIR.
    with open("docs/examples/Frameworks/pytorch_setup/job.sh") as f:
        job = f.read()

    assert "# conda create -y -n pytorch" in job
    assert "#     pytorch-cuda=11.6" in job
    assert "conda activate pytorch" in job

    to_replace = textwrap.dedent(
        """\
        # conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \\
        #     pytorch-cuda=11.6 -c pytorch -c nvidia
        """
    )
    assert to_replace in job
    job = job.replace(
        to_replace,
        textwrap.dedent(
            f"""\
            conda create -y -p {conda_env_prefix} python=3.9 pytorch torchvision torchaudio \\
                pytorch-cuda=11.6 -c pytorch -c nvidia
            """
        ),
    )
    assert "conda activate pytorch" in job
    job = job.replace("conda activate pytorch", f"conda activate {conda_env_prefix}")

    sbatch_script_path = Path(conda_dir / "setup_job.sh")
    sbatch_script_path.write_text(job)

    # TODO: Actually run the job that creates the conda environment.
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--output={conda_dir}/setup_job.out", str(sbatch_script_path)]
    )


def test_pytorch_setup(monkeypatch: pytest.MonkeyPatch, pytorch_conda_env: Path, tmp_path: Path):
    # monkeypatch.chdir("docs/examples/001_pytorch_setup")
    assert False, pytorch_conda_env
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--output={tmp_path}/job.out", "job.sh"]
    )
    assert False, job_id
