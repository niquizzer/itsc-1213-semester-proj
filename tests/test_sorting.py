"""Tests for analysis.sorting: custom sort keys and the stable frame sort."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis import sorting
from analysis.preconditions import PreconditionError


def stats_frame() -> pd.DataFrame:
    """A small theme-stats-like frame with ties, a NaN, and every class label."""
    return pd.DataFrame({
        "theme": [3, 0, 1, 2, 4, 5],
        "count": [40, 80, 80, 10, 55, 60],
        "high_pct": [0.05, 0.30, 0.30, math.nan, 0.12, 0.09],
        "classification": ["LOW", "HIGH", "HIGH", "Excluded (small)", "baseline", "baseline"],
        "residual": [-0.20, 0.15, 0.05, math.nan, 0.00, -0.01],
        "flag": ["oddball down", "oddball up", "oddball up", "Excluded (small)",
                 "not an oddball", "not an oddball"],
    })


# --- field access -----------------------------------------------------------


def test_field_reads_dicts_and_namedtuples() -> None:
    """One accessor works for dict rows and itertuples rows alike."""
    frame = pd.DataFrame({"theme": [7], "high_pct": [0.5]})
    row = next(frame.itertuples(index=False))
    assert sorting.field({"theme": 7}, "theme") == 7
    assert sorting.field(row, "theme") == 7
    assert sorting.field(row, "high_pct") == pytest.approx(0.5)


def test_field_rejects_missing_names() -> None:
    """A missing key or attribute is a precondition failure, not a KeyError."""
    with pytest.raises(PreconditionError):
        sorting.field({"theme": 1}, "nope")
    with pytest.raises(PreconditionError):
        sorting.field(object(), "theme")
    with pytest.raises(PreconditionError):
        sorting.field({"theme": 1}, "")


# --- numeric helpers --------------------------------------------------------


def test_ascending_and_descending_put_nan_last() -> None:
    """NaN maps to +inf under both helpers so it always sorts to the end."""
    values = [0.3, math.nan, -1.0, 2.0]
    asc = sorted(values, key=sorting.ascending)
    desc = sorted(values, key=sorting.descending)
    assert asc[:3] == [-1.0, 0.3, 2.0] and math.isnan(asc[-1])
    assert desc[:3] == [2.0, 0.3, -1.0] and math.isnan(desc[-1])


def test_rank_of_known_and_unknown_labels() -> None:
    """Known labels get their table rank; anything else sorts last."""
    assert sorting.rank_of("HIGH", sorting.CLASSIFICATION_RANK) == 0
    assert sorting.rank_of("LOW", sorting.CLASSIFICATION_RANK) == 2
    assert sorting.rank_of("mystery", sorting.CLASSIFICATION_RANK) == sorting.UNKNOWN_RANK
    assert sorting.rank_of("oddball up", sorting.ODDBALL_FLAG_RANK) == 0


# --- record keys ------------------------------------------------------------


def test_theme_id_key_orders_by_theme() -> None:
    """The canonical table order is ascending theme id."""
    ordered = sorting.sort_frame(stats_frame(), sorting.theme_id_key)
    assert ordered["theme"].tolist() == [0, 1, 2, 3, 4, 5]


def test_high_pct_keys_break_ties_by_theme_and_put_nan_last() -> None:
    """Ascending and descending high-rate orders, ties by theme id, NaN last."""
    frame = stats_frame()
    asc = sorting.sort_frame(frame, sorting.high_pct_key)["theme"].tolist()
    desc = sorting.sort_frame(frame, sorting.high_pct_desc_key)["theme"].tolist()
    assert asc == [3, 5, 4, 0, 1, 2]
    assert desc == [0, 1, 4, 5, 3, 2]


def test_residual_keys() -> None:
    """Residual ascending / descending / magnitude orders come out right."""
    frame = stats_frame()
    asc = sorting.sort_frame(frame, sorting.residual_key)["theme"].tolist()
    desc = sorting.sort_frame(frame, sorting.residual_desc_key)["theme"].tolist()
    magnitude = sorting.sort_frame(frame, sorting.residual_magnitude_key)["theme"].tolist()
    assert asc == [3, 5, 4, 1, 0, 2]
    assert desc == [0, 1, 4, 5, 3, 2]
    assert magnitude == [3, 0, 1, 5, 4, 2]


def test_count_desc_key() -> None:
    """Biggest theme first, ties by theme id."""
    ordered = sorting.sort_frame(stats_frame(), sorting.count_desc_key)
    assert ordered["theme"].tolist() == [0, 1, 5, 4, 3, 2]


def test_classification_rank_key_is_a_leaderboard() -> None:
    """HIGH first, then baseline, then LOW, then excluded; high-rate desc within."""
    ordered = sorting.sort_frame(stats_frame(), sorting.classification_rank_key)
    assert ordered["classification"].tolist() == [
        "HIGH", "HIGH", "baseline", "baseline", "LOW", "Excluded (small)",
    ]
    assert ordered["theme"].tolist() == [0, 1, 4, 5, 3, 2]


def test_oddball_flag_rank_key() -> None:
    """Up, then down, then not-oddball, then excluded; |residual| desc within."""
    ordered = sorting.sort_frame(stats_frame(), sorting.oddball_flag_rank_key)
    assert ordered["theme"].tolist() == [0, 1, 3, 5, 4, 2]


def test_genre_count_key_breaks_ties_alphabetically() -> None:
    """Most mentions first; equal counts fall back to alphabetical order."""
    makeup = pd.DataFrame({
        "theme": [0, 0, 0, 0],
        "genre": ["Drama", "Comedy", "Action", "Horror"],
        "count": [5, 7, 7, 1],
    })
    ordered = sorting.sort_frame(makeup, sorting.genre_count_key)
    assert ordered["genre"].tolist() == ["Action", "Comedy", "Drama", "Horror"]


# --- key factories ----------------------------------------------------------


def test_keyword_weight_key_ranks_heaviest_then_alphabetical() -> None:
    """Column indices sort by -mean weight, alphabetical among equal weights."""
    vocab = ["zeta", "alpha", "mid", "beta"]
    means = np.array([0.5, 0.5, 0.9, 0.1])
    order = sorted(range(len(vocab)), key=sorting.keyword_weight_key(means, vocab))
    assert [vocab[j] for j in order] == ["mid", "alpha", "zeta", "beta"]


def test_keyword_weight_key_preconditions() -> None:
    """Mismatched lengths, non-string vocab, and out-of-range columns all fail."""
    with pytest.raises(PreconditionError):
        sorting.keyword_weight_key([0.1, 0.2], ["only-one"])
    with pytest.raises(PreconditionError):
        sorting.keyword_weight_key([0.1], [42])
    key = sorting.keyword_weight_key([0.1], ["a"])
    with pytest.raises(PreconditionError):
        key(1)


def test_silhouette_key_prefers_smaller_k_on_ties() -> None:
    """Highest score wins; an exact tie resolves to the smaller k."""
    scores = {10: 0.20, 8: 0.25, 9: 0.25, 11: 0.10}
    assert max(scores, key=sorting.silhouette_key(scores)) == 8
    ranked = sorted(scores, key=sorting.silhouette_key(scores), reverse=True)
    assert ranked == [8, 9, 10, 11]


def test_silhouette_key_preconditions() -> None:
    """Empty tables and unknown k values are rejected."""
    with pytest.raises(PreconditionError):
        sorting.silhouette_key({})
    key = sorting.silhouette_key({2: 0.5})
    with pytest.raises(PreconditionError):
        key(3)


# --- sort_frame / sort_records ---------------------------------------------


def test_sort_frame_is_stable_and_keeps_index_labels() -> None:
    """Equal keys keep input order; index labels travel with their rows."""
    frame = pd.DataFrame({"theme": [5, 6, 7, 8], "count": [1, 2, 1, 2]}, index=[10, 11, 12, 13])
    ordered = sorting.sort_frame(frame, lambda row: (row.count,))
    assert ordered["theme"].tolist() == [5, 7, 6, 8]
    assert ordered.index.tolist() == [10, 12, 11, 13]


def test_sort_frame_reverse_and_empty() -> None:
    """reverse flips the order; an empty frame comes back empty with its columns."""
    frame = pd.DataFrame({"theme": [1, 2, 3]})
    assert sorting.sort_frame(frame, sorting.theme_id_key, reverse=True)["theme"].tolist() == [3, 2, 1]
    empty = sorting.sort_frame(frame.iloc[0:0], sorting.theme_id_key)
    assert empty.empty and list(empty.columns) == ["theme"]


def test_sort_frame_preconditions() -> None:
    """Non-frames, non-callables and non-bool reverse flags are rejected."""
    frame = pd.DataFrame({"theme": [1]})
    with pytest.raises(PreconditionError):
        sorting.sort_frame([1, 2, 3], sorting.theme_id_key)
    with pytest.raises(PreconditionError):
        sorting.sort_frame(frame, "theme")
    with pytest.raises(PreconditionError):
        sorting.sort_frame(frame, sorting.theme_id_key, reverse=1)


def test_sort_records_sorts_dicts_with_the_same_keys() -> None:
    """The same record keys work on plain lists of dicts."""
    records = [{"theme": 2, "high_pct": 0.1}, {"theme": 1, "high_pct": 0.4}]
    ordered = sorting.sort_records(records, sorting.high_pct_desc_key)
    assert [r["theme"] for r in ordered] == [1, 2]
    with pytest.raises(PreconditionError):
        sorting.sort_records(records, None)
