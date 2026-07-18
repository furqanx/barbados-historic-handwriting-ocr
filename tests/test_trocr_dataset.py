import pandas as pd
import torch
from PIL import Image

from src.data.trocr_dataset import (
    TrOCRCollate,
    TrOCRLineDataset,
    make_trocr_image_transform,
)


class FakeEncoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        max_length = max(len(text) for text in texts)
        ids = []
        for text in texts:
            encoded = [ord(char) % 20 + 1 for char in text]
            encoded.extend([self.pad_token_id] * (max_length - len(encoded)))
            ids.append(encoded)
        return FakeEncoded(torch.tensor(ids, dtype=torch.long))


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append(kwargs)
        batch = torch.zeros(len(images), 3, images[0].height, images[0].width)
        return {"pixel_values": batch}


def test_trocr_dataset_returns_image_and_normalized_text(tmp_path) -> None:
    image_path = tmp_path / "line-1.jpg"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    manifest = pd.DataFrame(
        [{"ID": "line-1", "image_path": str(image_path), "Target": "  A   B  "}]
    )

    sample = TrOCRLineDataset(manifest)[0]

    assert sample.image_id == "line-1"
    assert sample.image.size == (20, 10)
    assert sample.text == "A B"


def test_trocr_collate_encodes_images_and_labels() -> None:
    processor = FakeProcessor()
    samples = [
        type("Sample", (), {"image_id": "a", "image": Image.new("RGB", (8, 4)), "text": "AB"}),
        type("Sample", (), {"image_id": "b", "image": Image.new("RGB", (8, 4)), "text": "A"}),
    ]
    collate = TrOCRCollate(processor=processor)

    batch = collate(samples)

    assert batch.pixel_values.shape == (2, 3, 4, 8)
    assert batch.labels.shape == (2, 2)
    assert batch.labels[1, 1].item() == -100
    assert batch.image_ids == ["a", "b"]
    assert batch.texts == ["AB", "A"]
    assert processor.calls[0] == {"return_tensors": "pt"}


def test_trocr_aspect_collate_disables_processor_resize() -> None:
    processor = FakeProcessor()
    samples = [
        type("Sample", (), {"image_id": "a", "image": Image.new("RGB", (8, 4)), "text": None})
    ]
    collate = TrOCRCollate(processor=processor, preprocess_mode="aspect")

    batch = collate(samples)

    assert batch.labels is None
    assert processor.calls[0] == {"return_tensors": "pt", "do_resize": False}


def test_make_trocr_aspect_transform_returns_canvas() -> None:
    transform = make_trocr_image_transform(
        preprocess_mode="aspect",
        target_height=16,
        canvas_width=64,
    )
    image = Image.new("RGB", (20, 10), color="white")

    transformed = transform(image)

    assert transformed.size == (64, 16)
