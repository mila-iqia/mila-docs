"""List to stdout the files of the dataset"""

from pathlib import Path
import sys

import datasets

from py_utils import (
    get_dataset_builder, get_num_workers, get_raw_datasets, get_tokenizer,
    preprocess_datasets
)


if __name__ == "__main__":
    # Redirect outputs to stderr to avoid noize in stdout
    _stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        _CACHE_DIR = sys.argv[1]
    except IndexError:
        _CACHE_DIR = datasets.config.HF_DATASETS_CACHE
    try:
        _WORKERS = int(sys.argv[2])
    except IndexError:
        _WORKERS = get_num_workers()

    cache_dir = Path(_CACHE_DIR)
    builder = get_dataset_builder(cache_dir=_CACHE_DIR)
    raw_datasets = get_raw_datasets(builder)
    tokenizer = get_tokenizer()
    for dataset in preprocess_datasets(tokenizer, raw_datasets, num_workers=_WORKERS).values():
        for cache_file in dataset.cache_files:
            cache_file = Path(cache_file["filename"]).relative_to(cache_dir)
            print(cache_file, file=_stdout)
