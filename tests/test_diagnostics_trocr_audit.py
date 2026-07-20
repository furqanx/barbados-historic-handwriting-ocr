import pandas as pd
from PIL import Image

from src.constants import ID_COL, TARGET_COL
from src.diagnostics.trocr_audit import TrOCRTargetAuditConfig, trocr_target_audit_table


class FakeEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        special = 2 if add_special_tokens else 0
        return FakeEncoding(list(range(len(text) + special)))


def test_trocr_target_audit_flags_truncation_and_canvas_clipping(tmp_path) -> None:
    image = tmp_path / "wide.jpg"
    Image.new("RGB", (1000, 100), color="white").save(image)
    manifest = pd.DataFrame(
        [{ID_COL: "a", TARGET_COL: "A" * 10, "image_path": str(image)}]
    )
    config = TrOCRTargetAuditConfig(
        target_height=100,
        canvas_width=100,
        max_label_length=8,
        max_generation_length=8,
    )

    table = trocr_target_audit_table(
        manifest,
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert table["token_count"].item() == 12
    assert table["exceeds_max_label_length"].item() is True
    assert table["would_clip_aspect_canvas"].item() is True

