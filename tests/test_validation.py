"""Tests for stage 05-validate MovieLens validation (tests-first)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis import common

validate_mod = common.load_script("05-validate")


def themed_for_validation() -> pd.DataFrame:
    """Themed movies incl. an unlinked id (555) and one with no ratings (101)."""
    return pd.DataFrame({
        "id": [862, 8844, 100, 101, 103, 555],
        "title": [f"t{i}" for i in range(6)],
        "genres": [["Drama"]] * 6,
        "high_rating": [True, False, False, False, False, False],
        "theme": [0, 1, 0, 2, 3, 0],
        "vote_average": [7.7, 6.9, 6.0, 6.0, 6.0, 6.0],
    })


def test_validation_joins_links_and_ratings(
    toy_links_csv: Path, toy_ratings_csv: Path,
) -> None:
    """Per-theme mean user rating, rated-movie count, and rating count come out right."""
    themed = themed_for_validation()
    validation = validate_mod.validate_movielens(themed, toy_links_csv, toy_ratings_csv)
    by_theme = validation.set_index("theme")

    # theme 0 = movies 862 (ratings 4.0, 5.0) + 100 (3.0, 3.0); 555 has no link.
    assert by_theme.loc[0, "ml_rated_movies"] == 2
    assert by_theme.loc[0, "ml_n_ratings"] == 4
    assert by_theme.loc[0, "ml_mean_rating"] == pytest.approx(3.75)

    # theme 1 = movie 8844 (4.5, 4.5).
    assert by_theme.loc[1, "ml_rated_movies"] == 1
    assert by_theme.loc[1, "ml_mean_rating"] == pytest.approx(4.5)

    # theme 2 = movie 101: linked? no — no rating row either way, so zero coverage.
    assert by_theme.loc[2, "ml_rated_movies"] == 0
    assert by_theme.loc[2, "ml_n_ratings"] == 0
    assert by_theme.loc[2, "ml_mean_rating"] != by_theme.loc[2, "ml_mean_rating"]  # NaN

    # theme 3 = movie 103 (2.5).
    assert by_theme.loc[3, "ml_rated_movies"] == 1
    assert by_theme.loc[3, "ml_mean_rating"] == pytest.approx(2.5)


def test_validation_covers_every_theme(toy_links_csv: Path, toy_ratings_csv: Path) -> None:
    """Every theme in the themed set appears exactly once in the validation table."""
    themed = themed_for_validation()
    validation = validate_mod.validate_movielens(themed, toy_links_csv, toy_ratings_csv)
    assert set(validation["theme"]) == {0, 1, 2, 3}
    assert len(validation) == 4