from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_kraken_htr_experiments import _decode_profiles as kraken_decode_profiles
from scripts.run_pylaia_htr_experiments import _decode_profiles as pylaia_decode_profiles


def test_pylaia_lm_profile_uses_documented_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "pylaia-himanis"
    model_dir.mkdir()
    (model_dir / "language_model.arpa.gz").write_text("lm", encoding="utf-8")
    (model_dir / "tokens.txt").write_text("tokens", encoding="utf-8")
    (model_dir / "lexicon.txt").write_text("lexicon", encoding="utf-8")
    args = Namespace(
        decode_profile=["native", "lm"],
        profile_extra_arg=[],
        pylaia_lm_weight=1.5,
    )

    profiles = pylaia_decode_profiles(args, model_dir)

    assert [profile.name for profile in profiles] == ["native", "lm_w1p5"]
    lm_profile = profiles[1]
    assert lm_profile.already_detokenized is True
    assert "--decode.use_language_model" in lm_profile.extra_args
    assert str(model_dir / "language_model.arpa.gz") in lm_profile.extra_args


def test_pylaia_unknown_profile_requires_explicit_args(tmp_path: Path) -> None:
    args = Namespace(
        decode_profile=["beam25"],
        profile_extra_arg=[],
        pylaia_lm_weight=1.5,
    )

    with pytest.raises(ValueError, match="Unknown PyLaia decode profile"):
        pylaia_decode_profiles(args, tmp_path)


def test_kraken_unknown_profile_requires_explicit_args() -> None:
    args = Namespace(decode_profile=["beam"], profile_extra_arg=[])

    with pytest.raises(ValueError, match="Kraken does not expose"):
        kraken_decode_profiles(args)
