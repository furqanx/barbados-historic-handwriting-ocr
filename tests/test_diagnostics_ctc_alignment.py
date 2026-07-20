import pandas as pd
from PIL import Image

from src.constants import ID_COL, TARGET_COL
from src.diagnostics.ctc_alignment import (
    CTCAlignmentConfig,
    adjacent_repeat_count,
    ctc_alignment_table,
)


def test_adjacent_repeat_count_handles_ctc_repeat_requirement() -> None:
    assert adjacent_repeat_count("letter") == 1
    assert adjacent_repeat_count("***") == 2


def test_ctc_alignment_flags_too_short_time_axis(tmp_path) -> None:
    image = tmp_path / "line.jpg"
    Image.new("RGB", (100, 50), color="white").save(image)
    manifest = pd.DataFrame(
        [{ID_COL: "a", TARGET_COL: "A" * 40, "image_path": str(image)}]
    )

    table = ctc_alignment_table(
        manifest,
        config=CTCAlignmentConfig(target_height=50, max_width=100, time_downsample_factor=4),
    )

    assert table["encoder_time_steps"].item() == 25
    assert table["is_ctc_length_valid"].item() is False

