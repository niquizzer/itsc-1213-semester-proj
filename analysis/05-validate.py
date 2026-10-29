"""Stage 05: detect oddball themes and validate against MovieLens ratings.

An *oddball* is a theme whose share of high-rated movies differs from what
its members' own official genres would predict. Each movie's expected
high-probability is the mean of its genres' global high-rates; a theme's
residual is observed minus expected, and a bootstrap confidence interval (or
a fallback magnitude rule) decides whether the residual is a real surprise.

The MovieLens pass is an independent sanity check: per-theme mean *user*
rating joined via ``links_small.csv`` to ``ratings_small.csv``.

Every function documents its *preconditions* and enforces them with
:mod:`analysis.preconditions`; table ordering uses the named keys in
:mod:`analysis.sorting`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, save_json
from analysis.preconditions import (
    require,
    require_all,
    require_array_1d,
    require_columns,
    require_existing_file,
    require_int,
    require_non_empty,
    require_real,
    require_type,
)
from analysis.sorting import oddball_flag_rank_key, sort_frame, theme_id_key

#: Columns of the per-theme residual frame built by :func:`theme_residuals`.
RESIDUAL_COLUMNS: list[str] = ["theme", "count", "observed", "expected", "residual"]

#: Columns of the oddball frame built by :func:`flag_oddballs`.
ODDBALL_COLUMNS: list[str] = RESIDUAL_COLUMNS + ["flag", "ci_low", "ci_high"]

#: Columns of the MovieLens validation frame built by :func:`validate_movielens`.
VALIDATION_COLUMNS: list[str] = [
    "theme", "ml_mean_rating", "ml_rated_movies", "ml_n_ratings",
]


def _to_names(cell: object) -> list[str]:
    """Normalise a genres cell to a list of names (accepts lists and JSON strings).

    Preconditions:
        none - any value is accepted; anything that is not a list or a JSON
        list becomes ``[]``.
    """
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(cell, list):
        return cell
    return []


def _is_rate(value: object) -> bool:
    """True when ``value`` is a real number in ``[0, 1]`` (and not a ``bool``).

    Preconditions:
        none - any value is accepted.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        return False
    return 0.0 <= float(value) <= 1.0


def _require_rates(rates: object, name: str) -> None:
    """Require a mapping from genre name (str) to a rate in ``[0, 1]``.

    Preconditions:
        none beyond ``name`` being the argument's name for messages.
    """
    require_type(rates, Mapping, name)
    require_all(list(rates.keys()), lambda key: isinstance(key, str), name, "keyed by a str")
    require_all(list(rates.values()), _is_rate, name, "a rate in [0, 1]")


def genre_high_rates(themed: pd.DataFrame) -> dict[str, float]:
    """Global high-rated rate per official genre over the themed set.

    A movie contributes once to each of its distinct genres. Genres that
    never appear are simply absent from the result.

    Preconditions:
        ``themed`` is a ``DataFrame`` with columns ``genres`` and
        ``high_rating``.
    """
    require_columns(themed, ["genres", "high_rating"], "themed")
    high_tally: dict[str, int] = {}
    total_tally: dict[str, int] = {}
    for movie in themed.itertuples(index=False):
        for genre in set(_to_names(movie.genres)):
            total_tally[genre] = total_tally.get(genre, 0) + 1
            if bool(movie.high_rating):
                high_tally[genre] = high_tally.get(genre, 0) + 1
    return {
        genre: high_tally.get(genre, 0) / total
        for genre, total in total_tally.items()
    }


def expected_high_rate(themed: pd.DataFrame, rates: dict[str, float]) -> pd.Series:
    """Per-movie expected high probability from its own genres.

    The expectation is the plain mean of the movie's genre rates. Movies with
    no genres get ``NaN`` and are excluded from theme-level residuals
    downstream.

    Preconditions:
        ``themed`` is a ``DataFrame`` with a ``genres`` column; ``rates``
        maps genre name to a rate in ``[0, 1]`` and contains **every** genre
        that appears in ``themed`` (build it with :func:`genre_high_rates`
        over the same or a superset frame).
    """
    require_columns(themed, ["genres"], "themed")
    _require_rates(rates, "rates")
    unknown = {
        genre
        for movie in themed.itertuples(index=False)
        for genre in _to_names(movie.genres)
        if genre not in rates
    }
    require(not unknown, f"'rates' has no entry for genre(s) {sorted(unknown)}")

    expected: list[float] = []
    for movie in themed.itertuples(index=False):
        genres = _to_names(movie.genres)
        if genres:
            expected.append(float(np.mean([rates[g] for g in genres])))
        else:
            expected.append(float("nan"))
    return pd.Series(expected, index=themed.index, dtype=float)


def theme_residuals(themed: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Per-theme observed minus genre-expected high rate, in theme-id order.

    Preconditions:
        ``themed`` is a non-empty ``DataFrame`` with columns ``theme``,
        ``genres`` and ``high_rating``; ``rates`` satisfies the contract of
        :func:`expected_high_rate`.
    """
    require_columns(themed, ["theme", "genres", "high_rating"], "themed")
    require_non_empty(themed, "themed")
    expected = expected_high_rate(themed, rates)
    rows: list[dict[str, Any]] = []
    for theme, group in themed.groupby("theme"):
        member_expected = expected.loc[group.index].dropna()
        exp_rate = float(member_expected.mean()) if len(member_expected) else float("nan")
        obs_rate = float(group["high_rating"].mean())
        rows.append({
            "theme": int(theme),
            "count": len(group),
            "observed": obs_rate,
            "expected": exp_rate,
            "residual": obs_rate - exp_rate,
        })
    frame = pd.DataFrame(rows, columns=RESIDUAL_COLUMNS)
    return sort_frame(frame, theme_id_key).reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of ``values`` (fixed seed, so reproducible).

    Preconditions:
        ``values`` is a non-empty 1-d array (or list) of numbers; ``n_boot``
        is an ``int`` >= 1; ``seed`` is a non-negative ``int``.
    """
    require_array_1d(values, "values")
    require_non_empty(values, "values")
    require_int(n_boot, "n_boot", minimum=1)
    require_int(seed, "seed", minimum=0)
    sample_values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(sample_values, size=len(sample_values), replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def flag_oddballs(
    themed: pd.DataFrame,
    rates: dict[str, float],
    residuals: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
    fallback_magnitude: float = 0.10,
    min_size: int = 30,
) -> pd.DataFrame:
    """Flag themes whose residual is beyond its bootstrap CI or above a fallback magnitude.

    Decision order per theme: fewer than ``min_size`` movies -> ``"Excluded
    (small)"``; CI entirely above 0 -> ``"oddball up"``; entirely below 0 ->
    ``"oddball down"``; otherwise ``|mean residual| >= fallback_magnitude``
    decides the direction, else ``"not an oddball"``. Rows come back in
    theme-id order.

    Preconditions:
        ``themed`` is a ``DataFrame`` with columns ``theme``, ``genres`` and
        ``high_rating``; ``rates`` satisfies the contract of
        :func:`expected_high_rate`; ``residuals`` is a ``DataFrame`` with
        every column in :data:`RESIDUAL_COLUMNS` (as built by
        :func:`theme_residuals`); ``n_boot`` is an ``int`` >= 1; ``seed`` is
        a non-negative ``int``; ``fallback_magnitude`` is a non-negative real
        number; ``min_size`` is a non-negative ``int``.
    """
    require_columns(themed, ["theme", "genres", "high_rating"], "themed")
    require_columns(residuals, RESIDUAL_COLUMNS, "residuals")
    require_int(n_boot, "n_boot", minimum=1)
    require_int(seed, "seed", minimum=0)
    require_real(fallback_magnitude, "fallback_magnitude", minimum=0.0)
    require_int(min_size, "min_size", minimum=0)

    expected = expected_high_rate(themed, rates)
    rows: list[dict[str, Any]] = []
    for r in residuals.itertuples():
        theme = int(r.theme)
        ci_low = float("nan")
        ci_high = float("nan")
        if int(r.count) < min_size:
            flag = "Excluded (small)"
        else:
            members = themed[themed["theme"] == theme]
            resid_vals = np.asarray(
                [
                    float(high) - exp_val
                    for high, exp_val in zip(
                        members["high_rating"], expected.loc[members.index]
                    )
                    if not np.isnan(exp_val)
                ],
                dtype=float,
            )
            if len(resid_vals) == 0:
                flag = "not an oddball"
            else:
                ci_low, ci_high = bootstrap_mean_ci(resid_vals, n_boot, seed)
                mean = float(np.mean(resid_vals))
                if ci_low > 0.0:
                    flag = "oddball up"
                elif ci_high < 0.0:
                    flag = "oddball down"
                elif abs(mean) >= fallback_magnitude:
                    flag = "oddball up" if mean > 0.0 else "oddball down"
                else:
                    flag = "not an oddball"
        rows.append({
            "theme": theme,
            "count": int(r.count),
            "observed": float(r.observed),
            "expected": float(r.expected),
            "residual": float(r.residual),
            "flag": flag,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    frame = pd.DataFrame(rows, columns=ODDBALL_COLUMNS)
    return sort_frame(frame, theme_id_key).reset_index(drop=True)


def validate_movielens(
    themed: pd.DataFrame,
    links_path: Path,
    ratings_path: Path,
) -> pd.DataFrame:
    """Per-theme mean MovieLens user rating joined via the link files.

    Every theme in ``themed`` gets exactly one row, even when none of its
    movies has a MovieLens link or rating (then ``ml_mean_rating`` is ``NaN``
    and the counts are 0). Rows come back in theme-id order.

    Preconditions:
        ``themed`` is a ``DataFrame`` with columns ``id`` and ``theme``;
        ``links_path`` is an existing CSV with columns ``movieId`` and
        ``tmdbId``; ``ratings_path`` is an existing CSV with columns
        ``movieId`` and ``rating``.
    """
    require_columns(themed, ["id", "theme"], "themed")
    require_existing_file(links_path, "links_path")
    require_existing_file(ratings_path, "ratings_path")
    links = pd.read_csv(links_path, dtype=str, keep_default_na=False)
    ratings = pd.read_csv(ratings_path, dtype=str, keep_default_na=False)
    require_columns(links, ["movieId", "tmdbId"], "links csv")
    require_columns(ratings, ["movieId", "rating"], "ratings csv")
    records: dict[int, tuple[list[float], set[str]]] = {}

    for movie in themed.itertuples(index=False):
        theme = int(movie.theme)
        tmdb_id = str(int(movie.id))
        row = links[links["tmdbId"] == tmdb_id]
        records.setdefault(theme, ([], set()))
        if row.empty:
            continue
        movie_id = row.iloc[0]["movieId"]
        movie_ratings = ratings[ratings["movieId"] == movie_id]
        if not movie_ratings.empty:
            records[theme][1].add(movie_id)
            records[theme][0].extend(float(r) for r in movie_ratings["rating"])

    rows: list[dict[str, Any]] = []
    for theme in sorted(records):
        values, movie_ids = records[theme]
        rows.append({
            "theme": theme,
            "ml_mean_rating": float(np.mean(values)) if values else float("nan"),
            "ml_rated_movies": len(movie_ids),
            "ml_n_ratings": len(values),
        })
    frame = pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
    return sort_frame(frame, theme_id_key).reset_index(drop=True)


def main() -> None:
    """Run stage 05 end-to-end and write oddballs + validation artifacts.

    Preconditions:
        stage 03 has been run (``analysis/output/themed_set.csv`` exists) and
        ``data/raw/links_small.csv`` / ``data/raw/ratings_small.csv`` exist.
    """
    out = ensure_output_dir()
    require_existing_file(out / "themed_set.csv", "themed_set.csv")
    themed = pd.read_csv(out / "themed_set.csv")

    rates = genre_high_rates(themed)
    residuals = theme_residuals(themed, rates)
    flags = flag_oddballs(themed, rates, residuals)
    flags.to_csv(out / "oddballs.csv", index=False)
    save_json(rates, out / "genre_rates.json")

    links_path = PROJECT_ROOT / "data" / "raw" / "links_small.csv"
    ratings_path = PROJECT_ROOT / "data" / "raw" / "ratings_small.csv"
    validation = validate_movielens(themed, links_path, ratings_path)
    validation.to_csv(out / "validation.csv", index=False)

    print(f"global genre rates (top 5): {dict(list(rates.items())[:5])}")
    # Console order: oddballs up, then down, then the rest, biggest surprise
    # first within each group (the CSV stays in theme-id order).
    for row in sort_frame(flags, oddball_flag_rank_key).itertuples():
        print(f"  theme {row.theme:>2} [{'n/a' if row.flag == 'Excluded (small)' else row.flag:<14}] "
              f"n={row.count:>4} observed={row.observed:.3f} "
              f"expected={row.expected:.3f} residual={row.residual:+.3f}")
    print("MovieLens validation:")
    for row in validation.itertuples():
        print(f"  theme {row.theme:>2} ml_mean={row.ml_mean_rating:.2f} "
              f"rated_movies={row.ml_rated_movies} ratings={row.ml_n_ratings}")


if __name__ == "__main__":
    main()
