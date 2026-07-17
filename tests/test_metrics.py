from src.evaluation.metrics import normalize_whitespace, score_transcriptions


def test_normalize_whitespace_preserves_case_and_punctuation() -> None:
    assert normalize_whitespace("  Hello,   World!  ") == "Hello, World!"


def test_score_transcriptions_exact_match_is_zero() -> None:
    score = score_transcriptions(["Signed Sealed"], ["Signed Sealed"])

    assert score.wer == 0
    assert score.cer == 0
    assert score.score == 0


def test_score_transcriptions_detects_errors() -> None:
    score = score_transcriptions(["hello world"], ["hello word"])

    assert score.wer > 0
    assert score.cer > 0
    assert score.score > 0
