import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_imagenet():
    with tempfile.TemporaryDirectory() as tempdir:
        dataset_path = Path(tempdir) / "imagenet"
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            image.save(train_dir / f"image_{i}.png")
            image.save(val_dir / f"image_{i}.png")

        yield dataset_path
