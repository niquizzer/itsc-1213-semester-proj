"""Tests for stage 03-cluster: k selection, hard clustering, themes (tests-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from analysis import common

cluster_mod = common.load_script("03-cluster")

IDS = list(range(101, 113))  # 12 movies


def toy_matrix() -> sparse.csr_matrix:
    """12 rows in two separated groups (6 each) with a little within-group jitter."""
    rng = np.random.default_rng(0)
    group_a = np.array([[1, 1, 1, 0, 0, 0, 0, 0]] * 6, dtype=float)
    group_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 6, dtype=float)
    jitter = rng.normal(0.0, 0.08, size=(12, 8))
    return sparse.csr_matrix(np.vstack([group_a, group_b]) + jitter)


def toy_working() -> pd.DataFrame:
    """A minimal working-set frame around the 12 clustered ids plus an extra."""
    frames = []
    for movie_id in IDS:
        frames.append({
            "id": movie_id, "title": f"m{movie_id}", "genres": ["Drama"],
            "keywords": ["kw"], "vote_average": 7.0, "vote_count": 100.0,
            "high_rating": False,
        })
    frames.append({
        "id": 999, "title": "no-keywords", "genres": ["Drama"], "keywords": [],
        "vote_average": 5.0, "vote_count": 100.0, "high_rating": False,
    })
    return pd.DataFrame(frames)


def test_select_k_finds_true_cluster_count() -> None:
    """Silhouette picks k=2 for two well-separated groups."""
    matrix = toy_matrix()
    k, scores = cluster_mod.select_k(matrix, k_min=2, k_max=3, seed=42)
    assert k == 2
    assert set(scores) == {2, 3}
    assert scores[2] > scores[3]
    assert 0.0 <= scores[2] <= 1.0 + 1e-9


def test_fit_themes_assigns_every_movie_exactly_one_theme() -> None:
    """Hard clustering: labels are 0..k-1 and cover every row."""
    matrix = toy_matrix()
    labels = cluster_mod.fit_themes(matrix, k=2, seed=42)
    assert labels.shape == (12,)
    assert set(np.unique(labels)) == {0, 1}


def test_fit_themes_is_reproducible_with_fixed_seed() -> None:
    """Same seed -> identical labels."""
    matrix = toy_matrix()
    first = cluster_mod.fit_themes(matrix, k=2, seed=42)
    second = cluster_mod.fit_themes(matrix, k=2, seed=42)
    np.testing.assert_array_equal(first, second)


def test_build_themed_set_keeps_clusters_at_or_above_min_size() -> None:
    """Clusters >= min size are kept with a theme label and a summary."""
    matrix = toy_matrix()
    working = toy_working()
    labels = cluster_mod.fit_themes(matrix, k=2, seed=42)
    themed, summary = cluster_mod.build_themed_set(
        working, np.array(IDS, dtype=np.int64), labels, min_cluster_size=5,
    )
    assert set(themed["id"]) == set(IDS)          # 999 excluded (no keywords)
    assert 999 not in set(themed["id"])
    assert themed["theme"].nunique() == 2
    assert set(themed["theme"]) == {0, 1}
    assert summary["n_dropped"] == 0
    assert summary["min_cluster_size"] == 5


def test_build_themed_set_drops_small_clusters() -> None:
    """Clusters below min size are dropped and recorded in the summary."""
    matrix = toy_matrix()
    working = toy_working()
    labels = cluster_mod.fit_themes(matrix, k=2, seed=42)
    themed, summary = cluster_mod.build_themed_set(
        working, np.array(IDS, dtype=np.int64), labels, min_cluster_size=7,
    )
    assert themed.empty
    assert summary["n_dropped"] == 12
    assert len(summary["excluded_clusters"]) == 2
    assert summary["cluster_sizes"][0] == 6
    assert summary["cluster_sizes"][1] == 6


def test_build_themed_set_summary_reports_sizes() -> None:
    """Summary records per-cluster sizes and the kept movie count."""
    matrix = toy_matrix()
    working = toy_working()
    labels = cluster_mod.fit_themes(matrix, k=2, seed=42)
    themed, summary = cluster_mod.build_themed_set(
        working, np.array(IDS, dtype=np.int64), labels, min_cluster_size=5,
    )
    assert summary["n_movies_themed"] == 12
    assert len(summary["cluster_sizes"]) == 2
    assert sorted(summary["cluster_sizes"].values()) == [6, 6]