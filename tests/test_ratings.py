"""Tests for stage 04-ratings statistics: HIGH/LOW classification (tests-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import common

ratings_mod = common.load_script("04-ratings")


def test_theme_statistics_computes_count_mean_pct_lift(toy_themed_df: pd.DataFrame) -> None:
    """Per-theme count, mean rating, high percentage, and lift come out right."""
    stats = ratings_mod.theme_statistics(toy_themed_df, min_support=8, confidence=0.95)
    by_theme = stats.set_index("theme")

    assert by_theme.loc[0, "count"] == 8
    assert by_theme.loc[0, "mean_rating"] == pytest.approx(8.0)
    assert by_theme.loc[0, "high_pct"] == pytest.approx(1.0)
    assert by_theme.loc[0, "lift"] == pytest.approx(26.0 / 12.0)

    assert by_theme.loc[1, "count"] == 8
    assert by_theme.loc[1, "high_pct"] == pytest.approx(0.0)
    assert by_theme.loc[1, "mean_rating"] == pytest.approx(5.0)

    row2 = by_theme.loc[2]
    assert row2["count"] == 8
    assert row2["mean_rating"] == pytest.approx(6.125)
    assert row2["high_pct"] == pytest.approx(0.375)


def test_theme_statistics_reproduces_global_rate(toy_themed_df: pd.DataFrame) -> None:
    """The reported global high-rate matches the fixture's overall proportion."""
    stats = ratings_mod.theme_statistics(toy_themed_df, min_support=8, confidence=0.95)
    global_rate = float(stats["global_rate"].iloc[0])
    assert global_rate == pytest.approx(12.0 / 26.0)


def test_classification_high_low_baseline_and_excluded(toy_themed_df: pd.DataFrame) -> None:
    """Every theme is classified or explicitly excluded by the significance rule."""
    stats = ratings_mod.theme_statistics(toy_themed_df, min_support=8, confidence=0.95)
    by_theme = stats.set_index("theme")
    assert by_theme.loc[0, "classification"] == "HIGH"
    assert by_theme.loc[1, "classification"] == "LOW"
    assert by_theme.loc[2, "classification"] == "baseline"
    assert by_theme.loc[3, "classification"] == "Excluded (small)"


def test_all_themes_have_a_classification(toy_themed_df: pd.DataFrame) -> None:
    """No theme gets a missing/blank classification."""
    stats = ratings_mod.theme_statistics(toy_themed_df, min_support=8, confidence=0.95)
    assert not stats["classification"].isna().any()
    assert set(stats["theme"]) == {0, 1, 2, 3}


def test_theme_statistics_excludes_below_min_support(toy_themed_df: pd.DataFrame) -> None:
    """With a higher floor, themes below minimum support are excluded."""
    stats = ratings_mod.theme_statistics(toy_themed_df, min_support=9, confidence=0.95)
    excluded = stats[stats["classification"] == "Excluded (small)"]["theme"].tolist()
    assert 3 in excluded


def test_floor_sensitivity_global_rates() -> None:
    """Global high-rate at each vote floor, counting only movies at/above it."""
    meta = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "vote_average": [8.0, 7.5, 6.0, 7.49],
        "vote_count": [120.0, 60.0, 40.0, 150.0],
    })
    sensitivity = ratings_mod.floor_sensitivity(meta, floors=(25, 50, 100))
    rates = sensitivity.set_index("floor")
    assert rates.loc[25, "count"] == 4
    assert rates.loc[25, "high_rate"] == pytest.approx(0.5)
    assert rates.loc[50, "count"] == 3
    assert rates.loc[50, "high_rate"] == pytest.approx(2.0 / 3.0)
    assert rates.loc[100, "count"] == 2
    assert rates.loc[100, "high_rate"] == pytest.approx(0.5)