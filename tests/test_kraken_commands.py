from argparse import Namespace
from pathlib import Path

from scripts.train_kraken import build_train_command


def test_train_kraken_command_uses_explicit_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_files = dataset / "train_files.txt"
    val_files = dataset / "val_files.txt"
    model = tmp_path / "base.mlmodel"
    train_files.write_text("train/a.png\n", encoding="utf-8")
    val_files.write_text("val/b.png\n", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    args = Namespace(
        dataset_dir=dataset,
        base_model=model,
        run_name="kraken_test",
        output_prefix=tmp_path / "out" / "kraken_test",
        epochs=2,
        min_epochs=1,
        lag=1,
        batch_size=4,
        lr=0.001,
        weight_decay=0.0,
        optimizer="AdamW",
        resize="new",
        unicode_normalization="NFD",
        device="cpu",
        precision="32-true",
        workers=1,
        quit="fixed",
        augment=False,
        extra_arg=[],
    )

    command = build_train_command(args)

    assert command[:2] == ["ketos", "--workers"]
    assert "-t" in command
    assert str(train_files) in command
    assert "-e" in command
    assert str(val_files) in command
    assert "--no-augment" in command

