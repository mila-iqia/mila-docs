"""Dataset preprocessing script.

Run this with `srun --ntasks-per-node=1 --pty uv run python prepare_data.py`
"""

import argparse
import datetime
import os
from typing import Literal
from torchvision.datasets import ImageNet
from pathlib import Path

SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
NETWORK_IMAGENET_DIR = Path("/network/datasets/imagenet")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=SLURM_TMPDIR / "data",
        help="Where to prepare the dataset.",
    )
    parser.add_argument(
        "--network-imagenet-dir",
        type=Path,
        default=NETWORK_IMAGENET_DIR,
        help="The path to the folder containing the ILSVRC2012 train and val archives and devkit.",
    )
    dest = parser.parse_args().dest
    assert isinstance(dest, Path)
    # to see it as soon as it happens in logs.
    # `srun` can keep output in a buffer for quite a while otherwise.
    print(f"Preparing ImageNet dataset in {dest}", flush=True)
    _, _ = prepare_imagenet(dest)
    print(f"Done preparing ImageNet dataset in {dest}")


def prepare_imagenet(output_directory: Path, network_imagenet_dir: Path = NETWORK_IMAGENET_DIR):
    devkit_archive = network_imagenet_dir / "ILSVRC2012_devkit_t12.tar.gz"
    train_archive = network_imagenet_dir / "ILSVRC2012_img_train.tar"
    val_archive = network_imagenet_dir / "ILSVRC2012_img_val.tar"
    checksums_file = network_imagenet_dir / "md5sums"
    if any(
        not p.exists()
        for p in (network_imagenet_dir, devkit_archive, train_archive, val_archive, checksums_file)
    ):
        raise FileNotFoundError(
            f"Could not find the ImageNet dataset archives at {network_imagenet_dir}! "
            "Adjust the location with the argument as needed. "
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    _make_symlink_in_dest(devkit_archive, output_directory)
    _make_symlink_in_dest(train_archive, output_directory)
    _make_symlink_in_dest(val_archive, output_directory)
    _make_symlink_in_dest(checksums_file, output_directory)

    train_dataset = _make_split(output_directory, "train")
    test_dataset = _make_split(output_directory, "val")
    return train_dataset, test_dataset


def _make_symlink_in_dest(file: Path, dest_dir: Path):
    if not (symlink_to_file := (dest_dir / file.name)).exists():
        symlink_to_file.symlink_to(file)
    return symlink_to_file


def _make_split(root: Path, split: Literal["train", "val"]):
    """Use the torchvision.datasets.ImageNet class constructor to prepare the data.

    There are faster ways of doing this with the `tarfile` package or fancy bash
    commands but this is simplest.
    """
    print(f"Preparing ImageNet {split} split in {root}", flush=True)
    t = datetime.datetime.now()
    d = ImageNet(root=str(root), split=split)
    print(f"Preparing ImageNet {split} split took {datetime.datetime.now() - t}")
    return d


if __name__ == "__main__":
    main()
