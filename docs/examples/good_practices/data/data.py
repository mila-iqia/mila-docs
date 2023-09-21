"""Make sure the data is available"""
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from torchvision.datasets import INaturalist


def link_file(src: Path, dest: Path) -> None:
    src.symlink_to(dest)


def link_files(src: Path, dest: Path, workers: int = 4) -> None:
    os.makedirs(dest, exist_ok=True)
    with Pool(processes=workers) as pool:
        for path, dnames, fnames in os.walk(str(src)):
            rel_path = Path(path).relative_to(src)
            fnames = map(lambda _f: rel_path / _f, fnames)
            dnames = map(lambda _d: rel_path / _d, dnames)
            for d in dnames:
                os.makedirs(str(dest / d), exist_ok=True)
            pool.starmap(
                link_file,
                [(src / _f, dest / _f) for _f in fnames]
            )


if __name__ == "__main__":
    src = Path(sys.argv[1])
    workers = int(sys.argv[2])
    # Referencing $SLURM_TMPDIR here instead of job.sh makes sure that the
    # environment variable will only be resolved on the worker node (i.e. not
    # referencing the $SLURM_TMPDIR of the master node)
    dest = Path(os.environ["SLURM_TMPDIR"]) / "dest"

    start_time = time.time()

    link_files(src, dest, workers)

    # Torchvision expects these names
    shutil.move(dest / "train.tar.gz", dest / "2021_train.tgz")
    shutil.move(dest / "val.tar.gz", dest / "2021_valid.tgz")

    INaturalist(root=dest, version="2021_train", download=True)
    INaturalist(root=dest, version="2021_valid", download=True)

    seconds_spent = time.time() - start_time

    print(f"Prepared data in {seconds_spent/60:.2f}m")
