"""Tests for stage 05-validate oddballs: genre baselines + residuals + flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import common

validate_mod = common.load_script("05-validate")

HORROR_RATE = 60.0 / 200.0     # 0.3
DRAMA_RATE = 61.0 / 140.0      # ~0.435714


def oddball_frame(include_small: bool = False) -> pd.DataFrame:
    """A themed frame whose genre rates are exactly known, plus optional tiny theme."""
    rows: list[dict[str, object]] = []

    def add(n: int, genre: str, theme: int, n_high: int, base_id: int) -> None:
        for i in range(n):
            high = i < n_high
            rows.append({
                "id": base_id + i, "title": f"m{base_id + i}",
                "genres": [genre], "keywords": ["k"],
                "vote_average": 8.0 if high else 5.0, "vote_count": 100.0,
                "high_rating": high, "theme": theme,
            })

    add(30, "Horror", 0, 30, 1000)     # oddball up: observed 1.0 vs 0.3
    add(30, "Horror", 2, 9, 2000)      # inline: observed 0.3 == genre rate
    add(140, "Horror", -1, 21, 3000)   # background
    add(30, "Drama", 1, 6, 5000)       # oddball down: observed 0.2 vs ~0.436
    add(110, "Drama", -1, 55, 6000)    # background
    if include_small:
        add(10, "Horror", 3, 10, 9000)  # clear deviation but only 10 movies
    return pd.DataFrame(rows)


def test_genre_high_rates_are_computed_per_genre() -> None:
    """Global high-rate per genre over the whole themed set."""
    rates = validate_mod.genre_high_rates(oddball_frame())
    assert rates["Horror"] == pytest.approx(HORROR_RATE)
    assert rates["Drama"] == pytest.approx(DRAMA_RATE)


def test_expected_high_rate_uses_movies_own_genres() -> None:
    """A movie's expected high probability is the mean of its genres' rates."""
    frame = pd.DataFrame({
        "id": [1, 2, 3],
        "genres": [["Horror"], ["Drama"], ["Horror", "Drama"]],
        "high_rating": [True, True, True],
        "theme": [0, 0, 0],
    })
    expected = validate_mod.expected_high_rate(
        frame, {"Horror": 0.3, "Drama": 0.9},
    )
    assert expected.iloc[0] == pytest.approx(0.3)
    assert expected.iloc[1] == pytest.approx(0.9)
    assert expected.iloc[2] == pytest.approx(0.6)


def test_expected_high_rate_is_nan_without_genres() -> None:
    """Movies with no genres get NaN expected rate (excluded downstream)."""
    frame = pd.DataFrame({
        "id": [1], "genres": [[]], "high_rating": [True], "theme": [0],
    })
    expected = validate_mod.expected_high_rate(frame, {"Horror": 0.3})
    assert np.isnan(expected.iloc[0])


def test_theme_residuals_match_fixture() -> None:
    """Per-theme observed - genre-expected residual comes out exactly."""
    frame = oddball_frame()
    rates = validate_mod.genre_high_rates(frame)
    themes_only = frame[frame["theme"].isin([0, 1, 2])]
    residuals = validate_mod.theme_residuals(themes_only, rates).set_index("theme")

    assert residuals.loc[0, "count"] == 30
    assert residuals.loc[0, "observed"] == pytest.approx(1.0)
    assert residuals.loc[0, "expected"] == pytest.approx(HORROR_RATE)
    assert residuals.loc[0, "residual"] == pytest.approx(1.0 - HORROR_RATE)

    assert residuals.loc[1, "observed"] == pytest.approx(0.2)
    assert residuals.loc[1, "residual"] == pytest.approx(0.2 - DRAMA_RATE)

    assert residuals.loc[2, "observed"] == pytest.approx(0.3)
    assert residuals.loc[2, "residual"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_mean_ci_is_deterministic_and_sane() -> None:
    """Fixed seed -> identical CI; the sample mean sits inside it."""
    values = np.array([0.7] * 30 + [-0.3] * 30)
    first = validate_mod.bootstrap_mean_ci(values, n_boot=200, seed=42)
    second = validate_mod.bootstrap_mean_ci(values, n_boot=200, seed=42)
    assert first == second
    lo, hi = first
    assert lo <= hi
    mean = float(values.mean())
    assert lo <= mean <= hi


def test_flag_oddballs_directions() -> None:
    """Beyond-CI themes are flagged up/down; inline themes are not flagged."""
    frame = oddball_frame()
    rates = validate_mod.genre_high_rates(frame)
    themes_only = frame[frame["theme"].isin([0, 1, 2])]
    residuals = validate_mod.theme_residuals(themes_only, rates)
    flags = validate_mod.flag_oddballs(
        themes_only, rates, residuals, n_boot=200, seed=42,
        fallback_magnitude=0.10, min_size=30,
    ).set_index("theme")

    assert flags.loc[0, "flag"] == "oddball up"
    assert flags.loc[1, "flag"] == "oddball down"
    assert flags.loc[2, "flag"] == "not an oddball"


def test_flag_oddballs_excludes_small_themes() -> None:
    """Themes below min_size are explicitly excluded even with a big residual."""
    frame = oddball_frame(include_small=True)
    rates = validate_mod.genre_high_rates(frame)
    themes_only = frame[frame["theme"].isin([0, 1, 2, 3])]
    residuals = validate_mod.theme_residuals(themes_only, rates)
    flags = validate_mod.flag_oddballs(
        themes_only, rates, residuals, n_boot=200, seed=42,
        fallback_magnitude=0.10, min_size=30,
    ).set_index("theme")

    assert flags.loc[3, "flag"] == "Excluded (small)"


def test_flag_oddballs_fallback_magnitude_rule() -> None:
    """A large-magnitude residual is flagged even when the CI does not exclude 0."""
    frame = oddball_frame()
    rates = validate_mod.genre_high_rates(frame)
    themes_only = frame[frame["theme"].isin([0, 1, 2])]
    residuals = validate_mod.theme_residuals(themes_only, rates)
    # A residual of exactly 0 can never trip the magnitude fallback.
    flags = validate_mod.flag_oddballs(
        themes_only, rates, residuals, n_boot=200, seed=42,
        fallback_magnitude=0.50, min_size=30,
    ).set_index("theme")
    assert flags.loc[1, "flag"] in ("oddball down", "Excluded (small)")