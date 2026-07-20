from pathlib import Path

import pandas as pd
from PIL import Image

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.htr.kraken.dataset import prepare_kraken_dataset


def _make_image(path: Path, size: tuple[int, int] = (40, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_prepare_kraken_dataset_writes_line_strip_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    train_a = raw_dir / "a.jpg"
    train_b = raw_dir / "b.jpg"
    test_c = raw_dir / "c.jpg"
    _make_image(train_a)
    _make_image(train_b)
    _make_image(test_c)
    train_manifest = pd.DataFrame(
        [
            {ID_COL: "a", TARGET_COL: "By this", "image_path": str(train_a), FOLD_COL: 0},
            {ID_COL: "b", TARGET_COL: "Act", "image_path": str(train_b), FOLD_COL: 1},
        ]
    )
    test_manifest = pd.DataFrame([{ID_COL: "c", "image_path": str(test_c)}])

    paths = prepare_kraken_dataset(
        train_manifest,
        test_manifest,
        output_dir=tmp_path / "kraken",
        fold=0,
        image_height=16,
    )

    assert (paths.train_dir / "b.png").exists()
    assert (paths.train_dir / "b.gt.txt").read_text(encoding="utf-8") == "Act\n"
    assert (paths.val_dir / "a.png").exists()
    assert (paths.val_dir / "a.gt.txt").read_text(encoding="utf-8") == "By this\n"
    assert (paths.test_dir / "c.png").exists()
    assert (paths.test_dir / "c.gt.txt").exists() is False
    assert paths.test_files.read_text(encoding="utf-8").strip().endswith("c.png")

