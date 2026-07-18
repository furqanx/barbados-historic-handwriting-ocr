import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.data.char_tokenizer import build_tokenizer_from_texts
from src.data.ctc_dataset import CTCLineDataset, CTCCollate, make_ctc_image_transform


def test_ctc_dataset_returns_encoded_target_and_image_tensor(tmp_path) -> None:
    image_path = tmp_path / "line-1.jpg"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    manifest = pd.DataFrame(
        [{"ID": "line-1", "image_path": str(image_path), "Target": "A B"}]
    )
    tokenizer = build_tokenizer_from_texts(["A B"])
    transform = make_ctc_image_transform(target_height=8, max_width=None)

    dataset = CTCLineDataset(manifest, tokenizer, image_transform=transform)
    sample = dataset[0]

    assert sample.image_id == "line-1"
    assert sample.image.shape == (3, 8, 16)
    assert sample.text == "A B"
    assert sample.target_length == 3
    assert sample.target_ids.tolist() == tokenizer.encode("A B")


def test_ctc_collate_pads_width_and_flattens_targets(tmp_path) -> None:
    image_1 = tmp_path / "line-1.jpg"
    image_2 = tmp_path / "line-2.jpg"
    Image.new("RGB", (20, 10), color="white").save(image_1)
    Image.new("RGB", (30, 10), color="white").save(image_2)
    manifest = pd.DataFrame(
        [
            {"ID": "line-1", "image_path": str(image_1), "Target": "AB"},
            {"ID": "line-2", "image_path": str(image_2), "Target": "A"},
        ]
    )
    tokenizer = build_tokenizer_from_texts(["AB", "A"])
    transform = make_ctc_image_transform(target_height=8, max_width=None)
    dataset = CTCLineDataset(manifest, tokenizer, image_transform=transform)
    collate = CTCCollate(pad_value=1.0, width_multiple=8, time_downsample_factor=4)

    batch = collate([dataset[0], dataset[1]])

    assert batch.images.shape == (2, 3, 8, 24)
    assert batch.image_widths.tolist() == [16, 24]
    assert batch.input_lengths.tolist() == [4, 6]
    assert batch.target_lengths.tolist() == [2, 1]
    assert batch.targets.tolist() == tokenizer.encode("AB") + tokenizer.encode("A")
    assert batch.image_ids == ["line-1", "line-2"]
    assert batch.texts == ["AB", "A"]
    assert torch.all(batch.images[0, :, :, 16:] == 1.0)
