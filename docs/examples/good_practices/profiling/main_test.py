import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def prepare_imagenet():
    return None


@pytest.fixture(scope="function")
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
def setup_conda_environment(parse_requirements):
    """Create a conda environment following exactly the
    instructions in the docs and return the path to it."""
    requirements = parse_requirements

    # python_version =
    conda_env_dir: Path


# def test_conda_env_sees_gpu(setup_conda_environment):


@pytest.fixture(scope="session")
def path_to_conda_env():
    """Create a conda environment following exactly the instructions in the docs and return the path to it.

    TODO:
    - Read this a bit: https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html
    - Use this to create a temporary directory that will last the entire session: https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmp-path-factory-fixture
    - Create a conda environment with that directory as the prefix (with `conda create --prefix`) and the desired version of Python
    - pip install all the dependencies
    - return the path.
    """
    python_version = "3.10"  #
    conda_env_dir: Path = ...
    output = subprocess.run(
        f"conda create --yes --prefix {conda_env_dir} python={python_version}",
        text=True,
        capture_output=True,
        shell=True,
    )
    # then use the same idea to run `pip install` for all the dependencies
    ...  # TODO

    return conda_env_dir


@pytest.mark.xfail(reason="Not implemented yet")
## flag indicating that the test is expected to fail
def test_conda_env_sees_gpu(path_to_conda_env: Path):
    """Run something like this:

    ```bash
    conda activate {path_to_conda_env}
    python -c "import torch; print(torch.cuda.is_available())"
    ```
    """
    raise NotImplementedError


def test_run_example():
    path_to_conda_env = Path("/home/mila/c/cesar.valdez/venvs/docs")
    path_to_example = Path(__file__).parent / "main.py"
    result = subprocess.run(
        f"python {path_to_example} --epochs 1 --skip-training --n-samples 1000",
        # f"conda run -p {path_to_conda_env} python main.py --epochs 1 --skip-training --n-samples 1000",
        text=True,
        capture_output=True,
        shell=True,
    )
    if result.stdout:
        print("The example produced this output:")
        print(result.stdout)
    else:
        print("The example did not produce any output!")

    if result.stderr:
        print("The example produced this in stderr:")
        print(result.stderr)

    assert "accuracy:" in result.stdout

    # main("--epochs 1 --skip-training --num-samples 1000 ")
