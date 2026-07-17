from src.features.text_normalization import TextNormalizer, normalize_text


def test_default_normalization_preserves_case_punctuation_and_symbols() -> None:
    text = "  W^m   Olivant,  W^m Booke  "

    assert normalize_text(text) == "W^m Olivant, W^m Booke"


def test_optional_lowercase_and_punctuation_removal() -> None:
    normalizer = TextNormalizer(lowercase=True, remove_punctuation=True)

    assert normalizer("  W^m   Olivant,  ") == "wm olivant"


def test_none_becomes_empty_string() -> None:
    assert normalize_text(None) == ""
