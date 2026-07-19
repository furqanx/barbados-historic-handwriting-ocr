from pathlib import Path

from src.paddleocr_rec.text import (
    audit_paddleocr_characters,
    collect_paddleocr_characters,
    write_character_dict,
)


def test_collect_paddleocr_characters_excludes_space() -> None:
    characters = collect_paddleocr_characters(["A B"])

    assert characters == ["A", "B"]


def test_write_character_dict(tmp_path: Path) -> None:
    output = tmp_path / "dict.txt"

    write_character_dict(["A", "B"], output)

    assert output.read_text(encoding="utf-8") == "A\nB\n"


def test_audit_paddleocr_characters_keeps_space_in_audit() -> None:
    audit = audit_paddleocr_characters(["A B"])

    assert audit.has_space is True
    assert audit.character_count == 3

