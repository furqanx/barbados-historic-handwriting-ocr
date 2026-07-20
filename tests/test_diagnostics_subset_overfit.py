import pandas as pd

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.diagnostics.subset_overfit import make_overfit_manifest


def test_make_overfit_manifest_mirrors_same_samples_for_train_and_valid() -> None:
    manifest = pd.DataFrame(
        [
            {ID_COL: "a", TARGET_COL: "normal", FOLD_COL: 0},
            {ID_COL: "b", TARGET_COL: "rare ^", FOLD_COL: 1},
            {ID_COL: "c", TARGET_COL: "other", FOLD_COL: 2},
        ]
    )

    overfit = make_overfit_manifest(
        manifest,
        n_samples=2,
        require_chars="^",
        mirror_valid=True,
        seed=1,
    )

    assert len(overfit) == 4
    assert set(overfit[FOLD_COL]) == {0, 1}
    assert "b" in set(overfit[ID_COL])
    assert overfit[ID_COL].nunique() == 2
