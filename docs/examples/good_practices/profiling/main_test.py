import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

slurm_tmpdir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))


@pytest.fixture(scope="session")
def imagenet_dir():
    """Prepare the ImageNet dataset in the SLURM temporary directory."""
    _imagenet_dir = slurm_tmpdir / "imagenet"

    if not _imagenet_dir.exists():
        job_script_path = Path(__file__).parent / "make_imagenet.sh"
        subprocess.run(["bash", str(job_script_path)], check=True)

    return _imagenet_dir


def test_imagenet_preparation(imagenet_dir: Path):
    """Test that ImageNet data has been prepared correctly."""
    assert imagenet_dir.exists(), f"{imagenet_dir} does not exist"
    from torchvision.datasets import ImageNet

    # check that we can create the dataset and fetch an image
    ImageNet(imagenet_dir)[42]

    assert (
        imagenet_dir / "ILSVRC2012_img_train.tar"
    ).exists(), "Training data is missing"
    assert (
        imagenet_dir / "ILSVRC2012_img_val.tar"
    ).exists(), "Validation data is missing"


def parse_requirements():
    """
    Parse the requirements file and return a list of requirements.
    """

    def _parse(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()

        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

        return requirements

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    return _parse(requirements_file)


@pytest.fixture(scope="session")
def virtualenv():
    """
    Create a virtual environment at a temporary path with the
    requirements from the example.
    """
    requirements = parse_requirements()
    path_to_venv = slurm_tmpdir / "temp_env"

    if path_to_venv.exists():
        return path_to_venv

    create_venv = shlex.split(
        f"bash -c 'module load python/3.10 && python -m venv {path_to_venv}'"
    )
    subprocess.run(create_venv, check=True)

    pip_install_command = shlex.split(
        "bash -c '"
        "module load python/3.10 &&"
        f"source {path_to_venv}/bin/activate &&"
        f"pip install {' '.join(requirements)}"
        "'"
    )

    subprocess.run(pip_install_command, check=True)

    return Path(path_to_venv)  # returns path on succesful creation of conda env


def test_venv_sees_gpu(virtualenv: Path):
    check_gpu = shlex.split(
        "bash -c '"
        "module load python/3.10 && "
        f"source {virtualenv}/bin/activate && "
        'python -c "import torch; print(torch.cuda.is_available())"'
        "'"
    )

    result = subprocess.run(check_gpu, capture_output=True, check=True, text=True)

    assert "True" in result.stdout.strip(), "GPU is not available in the conda env"


def test_run_example(virtualenv: Path):
    path_to_example = Path(__file__).parent / "main.py"

    result = shlex.split(
        "bash -c '"
        "module load python/3.10 && "
        "module load cuda/11.7 && "
        f"source {virtualenv}/bin/activate && "
        f"python {path_to_example} --epochs 1 --skip-training --n-samples 1000"
        "'"
    )

    result = subprocess.run(result, capture_output=True, check=True, text=True)

    if result.stdout:
        print("The example produced this output:")
        print(result.stdout)
    else:
        print("The example did not produce any output!")

    if result.stderr:
        print("The example produced this in stderr:")
        print(result.stderr)

    last_line = result.stdout.strip().split("\n")[-1]
    metrics = json.loads(last_line)

    assert "samples/s" in metrics
    assert "updates/s" in metrics
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
