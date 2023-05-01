"""Preprocess the dataset.
In this example, HuggingFace is used and the resulting dataset will be stored in
$HF_DATASETS_CACHE. It is preferable to set the datasets cache to a location in
$SCRATCH"""

from itertools import chain
import os

import datasets
import torch
from transformers import AutoConfig, AutoTokenizer

_MODEL = "facebook/opt-125m"
_TOKENIZER = "facebook/opt-125m"
def _DS_BUILDER(*args, **kwargs):
    # Staticly write arguments that change the dataset's cache hash to avoid
    # redoing preprocessing / duplicating the dataset
    return datasets.load_dataset_builder("the_pile", *args, **kwargs,
                                         subsets=["all"], version="0.0.0")


def get_dataset_builder(cache_dir:str=None, local_dataset:str=None, num_workers:int=None):
    if local_dataset:
        local_dataset_splits = local_dataset.split("/")
        dl_config = datasets.DownloadConfig(cache_dir=local_dataset)
        # 'datasets' does not allow to use a local storage for the datasets' files using
        # it's exposed API. Mocking the download func to for the usage of the local file
        dl_man = datasets.DownloadManager(download_config=dl_config)
        _download = dl_man.download
        def dl(url_or_urls, *args, **kwargs):
            import glob
            local_files = ["/".join(_f.split("/")[len(local_dataset_splits):])
                           for _f in glob.glob(f"{local_dataset}/**", recursive=True)]
            local_files.sort()
            if isinstance(url_or_urls, str):
                url_or_urls = [url_or_urls]

            # Replace all urls by local files if they can be found
            for v in (url_or_urls.values() if isinstance(url_or_urls, dict) else {".":url_or_urls}):
                for i, url in enumerate(v):
                    for lf in local_files:
                        if lf and url.endswith(lf):
                            v[i] = f"{local_dataset}/{lf}"
                            local_files.remove(lf)
                            break

            # Continue normal download process which should only checksum the local
            # files instead of downloading them
            return _download(url_or_urls, *args, **kwargs)
        dl_man.download = dl
    else:
        dl_config = None
        dl_man = None

    builder = _DS_BUILDER(cache_dir=cache_dir, download_config=dl_config)
    builder.download_and_prepare(dl_manager=dl_man, num_proc=num_workers)

    return builder


def get_dataset_cache_dir(cache_dir:str=None):
    return _DS_BUILDER(cache_dir=cache_dir).cache_dir


def get_raw_datasets(builder):
    return builder.as_dataset().with_format("torch")


def get_config():
    return AutoConfig.from_pretrained(_MODEL)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(_TOKENIZER)


def tokenize_datasets(tokenizer, raw_datasets, num_workers:int=None, overwrite_cache=False):
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    return raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
    )


def preprocess_datasets(tokenizer, raw_datasets, num_workers:int=None, overwrite_cache=False):
    tokenized_datasets = tokenize_datasets(tokenizer, raw_datasets, num_workers=num_workers)

    block_size = tokenizer.model_max_length
    if block_size > 512:
        block_size = 512
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model
        # supported it instead of this drop, you can customize this part to
        # your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=not overwrite_cache
    )


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()
