import numpy as np
import polars as pl
import pytest

from honeypot.analysis.ml.encoding import FeatureEncoder


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "words": [["a", "b"], ["a"], ["b", "c"], ["c"], []],
            "x": [1.0, 2.0, 3.0, 4.0, None],
            "y": [10.0, 0.0, 10.0, 0.0, 5.0],
            "z": [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


def _row_norms(matrix) -> np.ndarray:
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())


def test_several_numeric_blocks_get_their_own_slices(frame) -> None:
    encoder = FeatureEncoder(
        token_blocks=("words",),
        numeric_blocks={"first": ("x", "y"), "second": ("z",)},
    ).fit(frame)
    slices = encoder.block_slices()
    assert list(slices) == ["words", "first", "second"]
    assert slices["second"] == slice(5, 6)
    assert encoder.feature_names()[-3:] == ["first:x", "first:y", "second:z"]
    matrix = encoder.transform(frame)
    assert matrix.shape == (5, 6)
    # Equal weights: each non-empty block holds a third of the squared norm.
    first = matrix[0, slices["first"]]
    assert first.multiply(first).sum() == pytest.approx(1 / 3)


def test_rows_unit_length_unless_numeric_left_as_z_scores(frame) -> None:
    blocks = {"token_blocks": ("words",), "numeric_blocks": {"num": ("x", "y")}}
    normalized = FeatureEncoder(**blocks).fit(frame).transform(frame)
    assert _row_norms(normalized[:4]) == pytest.approx(np.ones(4))

    raw = FeatureEncoder(**blocks, normalize_rows=False).fit(frame).transform(frame)
    numeric = raw[:, -2:].toarray() / np.sqrt(0.5)
    assert numeric.mean(axis=0) == pytest.approx(np.zeros(2))
    assert numeric.std(axis=0) == pytest.approx(np.ones(2))


def test_weight_zero_drops_block_and_weights_share_the_norm(frame) -> None:
    encoder = FeatureEncoder(
        token_blocks=("words",),
        numeric_blocks={"num": ("x", "y")},
        weights={"words": 0},
    ).fit(frame)
    assert list(encoder.block_slices()) == ["num"]

    weighted = FeatureEncoder(
        token_blocks=("words",),
        numeric_blocks={"num": ("x", "y")},
        weights={"words": 3, "num": 1},
    ).fit(frame)
    words = weighted.transform(frame)[0, weighted.block_slices()["words"]]
    assert words.multiply(words).sum() == pytest.approx(0.75)


def test_min_df_drops_rare_tokens(frame) -> None:
    encoder = FeatureEncoder(token_blocks=("words",), numeric_blocks={}, min_df=2).fit(
        frame
    )
    assert encoder.feature_names() == ["words:a", "words:b", "words:c"]
    rare = pl.DataFrame({"words": [["a", "zzz"]]})
    assert encoder.transform(rare)[0, :].nnz == 1
