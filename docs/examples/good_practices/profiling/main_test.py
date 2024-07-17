import os
import shutil
import tempfile

from main import create_dataloader, make_datasets


def copy_tree(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def test_directory_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "imagenet/train"), exist_ok=True)
        copy_tree("/tmp/imagenet/train", os.path.join(temp_dir, "imagenet/train"))

        assert os.path.isdir(
            os.path.join(temp_dir, "imagenet/train")
        ), "Train directory does not exist"
        assert (
            len(os.listdir(os.path.join(temp_dir, "imagenet/train"))) > 0
        ), "Train directory is empty"


def test_make_datasets():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "imagenet/train"), exist_ok=True)
        copy_tree("/tmp/imagenet/train", os.path.join(temp_dir, "imagenet/train"))

        train_dataset, _ = make_datasets(os.path.join(temp_dir, "imagenet"))
        assert len(train_dataset) > 0, "Train dataset is empty"


def test_dataloader():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "imagenet/train"), exist_ok=True)
        copy_tree("/tmp/imagenet/train", os.path.join(temp_dir, "imagenet/train"))

        train_dataset, _ = make_datasets(os.path.join(temp_dir, "imagenet"))
        train_loader = create_dataloader(train_dataset, batch_size=32)

        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        assert images.size(0) == 32, "Batch size is incorrect"
        assert len(labels) == 32, "Labels size is incorrect"
