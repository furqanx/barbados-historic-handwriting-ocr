from src.kraken_htr.text import audit_characters, normalize_for_kraken


def test_normalize_for_kraken_supports_nfd() -> None:
    value = normalize_for_kraken("é", unicode_form="NFD")

    assert value == "e\u0301"


def test_audit_characters_after_normalization() -> None:
    audit = audit_characters(["A A", "é"], unicode_form="NFD")

    assert audit.character_count == 4
    assert "e" in audit.characters
    assert "\u0301" in audit.characters

