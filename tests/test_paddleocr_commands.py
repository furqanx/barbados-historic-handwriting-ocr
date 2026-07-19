from argparse import Namespace
from pathlib import Path

from scripts.train_paddleocr_rec import build_train_command


def test_train_paddleocr_command_supports_distributed(tmp_path: Path) -> None:
    paddleocr_dir = tmp_path / "PaddleOCR"
    train_py = paddleocr_dir / "tools" / "train.py"
    train_py.parent.mkdir(parents=True)
    train_py.write_text("print('train')", encoding="utf-8")
    config = tmp_path / "config.yml"
    config.write_text("Global: {}", encoding="utf-8")
    args = Namespace(
        paddleocr_dir=paddleocr_dir,
        config=config,
        gpus="0,1",
        dry_run=True,
        override=["Global.epoch_num=1"],
    )

    command = build_train_command(args)

    assert "-m" in command
    assert "paddle.distributed.launch" in command
    assert "--gpus" in command
    assert "0,1" in command
    assert "-o" in command
    assert "Global.epoch_num=1" in command

