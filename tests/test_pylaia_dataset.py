from pathlib import Path

import pandas as pd
from PIL import Image

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.htr.pylaia.dataset import prepare_pylaia_dataset


def _make_image(path: Path, size: tuple[int, int] = (40, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_prepare_pylaia_dataset_writes_expected_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "raw"
    train_a = image_dir / "a.jpg"
    train_b = image_dir / "b.jpg"
    test_c = image_dir / "c.jpg"
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

    paths = prepare_pylaia_dataset(
        train_manifest,
        test_manifest,
        output_dir=tmp_path / "pylaia",
        fold=0,
        image_height=16,
    )

    assert (paths.images_dir / "train" / "b.jpg").exists()
    assert (paths.images_dir / "val" / "a.jpg").exists()
    assert (paths.images_dir / "test" / "c.jpg").exists()
    assert paths.train_txt.read_text(encoding="utf-8").startswith("train/b A c t")
    assert "val/a By this" in paths.val_text.read_text(encoding="utf-8")
    assert paths.test_ids.read_text(encoding="utf-8") == "test/c\n"
    assert paths.syms.read_text(encoding="utf-8").splitlines()[0] == "<ctc> 0"

