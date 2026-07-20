from src.constants import ID_COL, TARGET_COL
from src.htr.pylaia.prediction_parser import parse_pylaia_decode_output


def test_parse_pylaia_decode_output_strips_split_prefix() -> None:
    raw = "test/MzQuRiUbPFsq6Azy B y <space> t h i s\n"

    parsed = parse_pylaia_decode_output(raw)

    assert parsed.to_dict("records") == [
        {ID_COL: "MzQuRiUbPFsq6Azy", TARGET_COL: "By this"}
    ]


def test_parse_empty_prediction_line() -> None:
    raw = "test/abc123\n"

    parsed = parse_pylaia_decode_output(raw)

    assert parsed.to_dict("records") == [{ID_COL: "abc123", TARGET_COL: ""}]

