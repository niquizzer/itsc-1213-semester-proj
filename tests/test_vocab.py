"""Tests for stage 02-vocab: keyword curation and TF-IDF (tests-first)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from analysis import common

vocab_mod = common.load_script("02-vocab")


def test_clean_keyword_list_lowercases_dedupes_and_strips() -> None:
    """Cleaning lowercases, strips, and de-duplicates, keeping first order."""
    raw = [" Jealousy ", "toy", "Boy", "friendship", "friends", "Toy", "jealousy", ""]
    assert vocab_mod.clean_keyword_list(raw) == [
        "jealousy", "toy", "boy", "friendship", "friends",
    ]
    assert vocab_mod.clean_keyword_list([]) == []


def test_tfidf_frequency_filters_apply(toy_keyword_lists: list[list[str]]) -> None:
    """Terms in <3 or >40% of documents are dropped; kept terms match the set."""
    lists = toy_keyword_lists
    ids = list(range(len(lists)))
    matrix, vocab, kept_ids, stats = vocab_mod.build_tfidf(
        lists, ids, min_df=3, max_df=0.4, max_features=1000,
    )
    assert set(vocab) == {"alpha", "beta", "zeta", "eta"}
    assert kept_ids == ids
    assert stats["excluded_ids"] == []
    assert matrix.shape == (len(lists), len(vocab)) == (8, 4)


def test_tfidf_max_features_caps_vocabulary(toy_keyword_lists: list[list[str]]) -> None:
    """max_features caps the vocabulary size at the requested bound."""
    lists = toy_keyword_lists
    matrix, vocab, _, stats = vocab_mod.build_tfidf(
        lists, list(range(len(lists))), min_df=1, max_df=1.0, max_features=2,
    )
    assert matrix.shape[1] == 2
    assert stats["vocab_size"] == 2


def test_tfidf_excludes_empty_docs_and_normalizes_rows(
    toy_keyword_lists: list[list[str]],
) -> None:
    """Docs with no keywords are excluded; remaining rows are L2-normalized."""
    lists = toy_keyword_lists + [[]]
    ids = list(range(100, 100 + len(lists)))
    matrix, vocab, kept_ids, stats = vocab_mod.build_tfidf(
        lists, ids, min_df=2, max_df=0.5, max_features=1000,
    )
    assert stats["n_excluded"] == 1
    assert stats["excluded_ids"] == [100 + len(toy_keyword_lists)]
    assert kept_ids == ids[:-1]
    assert matrix.shape[0] == len(toy_keyword_lists)
    row_norms = np.asarray((matrix.multiply(matrix)).sum(axis=1)).ravel()
    assert np.allclose(row_norms, 1.0, atol=1e-6)


def test_tfidf_returns_sparse_matrix(toy_keyword_lists: list[list[str]]) -> None:
    """The returned matrix is a scipy sparse matrix."""
    matrix, _, _, _ = vocab_mod.build_tfidf(
        toy_keyword_lists, list(range(len(toy_keyword_lists))),
        min_df=1, max_df=1.0, max_features=1000,
    )
    assert sparse.issparse(matrix)
    assert matrix.nnz > 0