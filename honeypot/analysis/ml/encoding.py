# honeypot/analysis/ml/encoding.py
#
# Rows of token lists and numbers -> one sparse feature matrix, shared by
# the ml experiments. Everything *learned* -- TF-IDF vocabularies and IDF
# weights, numeric scaling -- is fit on the training rows only and then
# applied to validation/test, like any sklearn transformer, so nothing
# about held-out rows leaks into the features.
#
# Blocks, each scaled to its share of the total weight so a block with
# many columns (credentials: thousands of pairs) doesn't drown out a
# small one (a dozen numbers):
#
# - token blocks: a list[str] column each, TF-IDF over tokens seen in at
#   least `min_df` training rows (rarer ones are dropped: they can't
#   generalize), sublinear tf;
# - numeric blocks: named groups of numeric columns, standardized (nulls
#   -> 0 before scaling).
#
# Each block's rows are L2-normalized and multiplied by
# sqrt(weight / total weight), so a row's full vector has unit length and
# each block contributes its weight's share of it (a row with nothing in
# a block gets zeros there). Weight 0 drops a block. TF-IDF rows are
# always unit length; `normalize_rows=False` keeps numeric blocks as plain
# z-scores instead (for models where a number's size matters on its own).
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

DEFAULT_MIN_DF = 2


def _identity(tokens: list[str]) -> list[str]:
    return tokens


@dataclass
class FeatureEncoder:
    """Frame -> sparse matrix (one row per frame row). `token_blocks` names
    list[str] columns, `numeric_blocks` maps a block name to its numeric
    columns; `weights` has one entry per block (missing = 1)."""

    token_blocks: tuple[str, ...]
    numeric_blocks: Mapping[str, tuple[str, ...]]
    weights: Mapping[str, float] = field(default_factory=dict)
    min_df: int = DEFAULT_MIN_DF
    normalize_rows: bool = True
    vectorizers: dict[str, TfidfVectorizer] = field(default_factory=dict, init=False)
    scalers: dict[str, StandardScaler] = field(default_factory=dict, init=False)

    def _weight(self, block: str) -> float:
        return self.weights.get(block, 1.0)

    def fit(self, frame: pl.DataFrame) -> "FeatureEncoder":
        for block in self.token_blocks:
            if self._weight(block) <= 0:
                continue
            vectorizer = TfidfVectorizer(
                analyzer=_identity,
                lowercase=False,
                min_df=self.min_df,
                sublinear_tf=True,
            )
            try:
                vectorizer.fit(frame[block].to_list())
            except ValueError:
                # No token reaches min_df: the block is empty for everyone.
                continue
            self.vectorizers[block] = vectorizer
        for block, columns in self.numeric_blocks.items():
            if self._weight(block) > 0:
                self.scalers[block] = StandardScaler().fit(
                    self._numeric(frame, columns)
                )
        return self

    @staticmethod
    def _numeric(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        return frame.select(columns).fill_null(0).to_numpy()

    def _scale(self, block: str) -> float:
        blocks = (*self.token_blocks, *self.numeric_blocks)
        total = sum(w for b in blocks if (w := self._weight(b)) > 0)
        return float(np.sqrt(self._weight(block) / total))

    def transform(self, frame: pl.DataFrame) -> sp.csr_matrix:
        parts = []
        for block, vectorizer in self.vectorizers.items():
            matrix = vectorizer.transform(frame[block].to_list())
            parts.append(matrix * self._scale(block))
        for block, scaler in self.scalers.items():
            scaled = scaler.transform(self._numeric(frame, self.numeric_blocks[block]))
            if self.normalize_rows:
                norms = np.linalg.norm(scaled, axis=1, keepdims=True)
                scaled = np.divide(
                    scaled, norms, out=np.zeros_like(scaled), where=norms > 0
                )
            parts.append(sp.csr_matrix(scaled * self._scale(block)))
        return sp.hstack(parts, format="csr")

    def feature_names(self) -> list[str]:
        names = [
            f"{block}:{token}"
            for block, vectorizer in self.vectorizers.items()
            for token in vectorizer.get_feature_names_out()
        ]
        for block in self.scalers:
            names += [f"{block}:{name}" for name in self.numeric_blocks[block]]
        return names

    def block_slices(self) -> dict[str, slice]:
        """Column range of each block in transform()'s output."""
        slices, start = {}, 0
        for block, vectorizer in self.vectorizers.items():
            width = len(vectorizer.vocabulary_)
            slices[block] = slice(start, start + width)
            start += width
        for block in self.scalers:
            width = len(self.numeric_blocks[block])
            slices[block] = slice(start, start + width)
            start += width
        return slices
