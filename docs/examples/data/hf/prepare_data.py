"""Preprocess the dataset.
In this example, HuggingFace is used and the resulting dataset will be stored in
$HF_DATASETS_CACHE. It is preferable to set the datasets cache to a location in
$SCRATCH"""

from py_utils import (
    get_config, get_dataset_builder, get_num_workers, get_raw_datasets,
    get_tokenizer, preprocess_datasets
)


if __name__ == "__main__":
    import sys
    import time

    _LOCAL_DS = sys.argv[1]
    try:
        _WORKERS = int(sys.argv[2])
    except IndexError:
        _WORKERS = get_num_workers()

    t = -time.time()
    _ = get_config()
    builder = get_dataset_builder(local_dataset=_LOCAL_DS, num_workers=_WORKERS)
    raw_datasets = get_raw_datasets(builder)
    tokenizer = get_tokenizer()
    _ = preprocess_datasets(tokenizer, raw_datasets, num_workers=_WORKERS)
    t += time.time()

    print(f"Prepared data in {t/60:.2f}m")
