"""Stage 05: detect oddball themes and validate against MovieLens ratings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, save_json


def _to_names(cell: object) -> list[str]:
    """Normalize a genres cell to a list of names (accepts lists and JSON)."""
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(cell, list):
        return cell
    return []


def genre_high_rates(themed: pd.DataFrame) -> dict[str, float]:
    """Global high-rated rate per official genre over the themed set."""
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
    """Per-movie expected high probability from its own genres."""
    expected: list[float] = []
    for movie in themed.itertuples(index=False):
        genres = _to_names(movie.genres)
        if genres:
            expected.append(float(np.mean([rates[g] for g in genres])))
        else:
            expected.append(float("nan"))
    return pd.Series(expected, index=themed.index, dtype=float)


def theme_residuals(themed: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Per-theme observed minus genre-expected high rate."""
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
    return pd.DataFrame(rows).sort_values("theme").reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of values (fixed seed)."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
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
    """Flag themes whose residual is beyond CI or above a fallback magnitude."""
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
    return pd.DataFrame(rows).sort_values("theme").reset_index(drop=True)


def validate_movielens(
    themed: pd.DataFrame,
    links_path: Path,
    ratings_path: Path,
) -> pd.DataFrame:
    """Per-theme mean MovieLens user rating joined via the link files."""
    links = pd.read_csv(links_path, dtype=str, keep_default_na=False)
    ratings = pd.read_csv(ratings_path, dtype=str, keep_default_na=False)
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
    return pd.DataFrame(rows)


def main() -> None:
    """Run stage 05 end-to-end and write oddballs + validation artifacts."""
    out = ensure_output_dir()
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
    for row in flags.itertuples():
        print(f"  theme {row.theme:>2} [{'n/a' if row.flag == 'Excluded (small)' else row.flag:<14}] "
              f"n={row.count:>4} observed={row.observed:.3f} "
              f"expected={row.expected:.3f} residual={row.residual:+.3f}")
    print("MovieLens validation:")
    for row in validation.itertuples():
        print(f"  theme {row.theme:>2} ml_mean={row.ml_mean_rating:.2f} "
              f"rated_movies={row.ml_rated_movies} ratings={row.ml_n_ratings}")


if __name__ == "__main__":
    main()