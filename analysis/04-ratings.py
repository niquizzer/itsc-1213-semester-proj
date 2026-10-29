"""Stage 04: label themes and compute per-theme rating outcomes.

Two jobs: (1) *label* each theme by its top TF-IDF keywords and its official
genre make-up so it reads as a sentence, and (2) compute each theme's rating
statistics (size, mean rating, share of high-rated movies, lift over the
global rate) and classify it HIGH / LOW / baseline with a normal-approximation
confidence interval.

Ordering rules live in :mod:`analysis.sorting` as named key functions
(``keyword_weight_key``, ``genre_count_key``, ``theme_id_key``,
``classification_rank_key``) instead of ad-hoc lambdas, and every function
documents and enforces its *preconditions* via :mod:`analysis.preconditions`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, load_json, load_sparse, save_json
from analysis.preconditions import (
    require,
    require_all,
    require_array_1d,
    require_columns,
    require_existing_file,
    require_fraction,
    require_int,
    require_non_empty,
    require_sparse_matrix,
    require_type,
)
from analysis.sorting import (
    classification_rank_key,
    genre_count_key,
    keyword_weight_key,
    sort_frame,
    theme_id_key,
)

HIGH_THRESHOLD: float = 7.5

#: Columns of the per-theme genre make-up frame built by :func:`theme_genre_makeup`.
GENRE_MAKEUP_COLUMNS: list[str] = ["theme", "genre", "count", "share"]

#: Columns of the per-theme statistics frame built by :func:`theme_statistics`.
THEME_STATS_COLUMNS: list[str] = [
    "theme", "count", "mean_rating", "high_pct", "lift", "classification", "global_rate",
]


def _to_names(cell: object) -> list[str]:
    """Normalise a genres cell to a list of names (accepts lists and JSON strings).

    The themed set holds genres as Python lists in memory but as JSON strings
    once it has been round-tripped through CSV; both must work.

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


def theme_top_keywords(
    matrix: sparse.csr_matrix,
    vocab: list[str],
    tfidf_ids: np.ndarray,
    themed: pd.DataFrame,
    top_n: int,
) -> dict[int, list[str]]:
    """Return each theme's ``top_n`` characteristic keywords by mean TF-IDF.

    For every theme, the mean TF-IDF weight of each vocabulary term across the
    theme's member rows is computed; terms are then ranked with
    :func:`analysis.sorting.keyword_weight_key` - heaviest first, ties broken
    alphabetically - and the first ``top_n`` are returned.

    Preconditions:
        ``matrix`` is a 2-d SciPy sparse matrix; ``vocab`` is a ``list`` with
        exactly one entry per matrix column; ``tfidf_ids`` is a 1-d array
        (or list) with exactly one movie id per matrix row; ``themed`` is a
        ``DataFrame`` with columns ``id`` and ``theme``; ``top_n`` is an
        ``int`` >= 1.
    """
    require_sparse_matrix(matrix, "matrix")
    require_type(vocab, list, "vocab")
    require(
        len(vocab) == matrix.shape[1],
        f"'vocab' has {len(vocab)} terms but 'matrix' has {matrix.shape[1]} columns",
    )
    require_array_1d(tfidf_ids, "tfidf_ids")
    require(
        len(tfidf_ids) == matrix.shape[0],
        f"'tfidf_ids' has {len(tfidf_ids)} ids but 'matrix' has {matrix.shape[0]} rows",
    )
    require_columns(themed, ["id", "theme"], "themed")
    require_int(top_n, "top_n", minimum=1)

    row_of = {int(movie_id): row for row, movie_id in enumerate(tfidf_ids)}
    members: dict[int, list[int]] = {}
    for movie in themed.itertuples(index=False):
        row = row_of.get(int(movie.id))
        if row is not None:
            members.setdefault(int(movie.theme), []).append(row)
    result: dict[int, list[str]] = {}
    for theme in sorted(members):
        rows = members[theme]
        means = np.asarray(matrix[rows].mean(axis=0)).ravel()
        # Custom key: rank column indices by (-mean weight, keyword text).
        order = sorted(range(len(vocab)), key=keyword_weight_key(means, vocab))
        result[theme] = [vocab[j] for j in order[:top_n]]
    return result


def theme_genre_makeup(themed: pd.DataFrame) -> pd.DataFrame:
    """Return per-theme genre mention counts and within-theme shares as a frame.

    A movie with three genres contributes one mention to each; ``share`` is
    the mention count divided by the theme's total mentions, so shares within
    a theme sum to 1. Movies with no genres contribute nothing.

    Preconditions:
        ``themed`` is a ``DataFrame`` with columns ``theme`` and ``genres``
        (lists or JSON-encoded lists of genre names).
    """
    require_columns(themed, ["theme", "genres"], "themed")
    counts: dict[int, dict[str, int]] = {}
    for movie in themed.itertuples(index=False):
        theme = int(movie.theme)
        for genre in _to_names(movie.genres):
            counts.setdefault(theme, {}).setdefault(genre, 0)
            counts[theme][genre] += 1
    rows: list[dict[str, Any]] = []
    for theme, genre_counts in counts.items():
        total = sum(genre_counts.values())
        for genre, count in genre_counts.items():
            rows.append({
                "theme": theme,
                "genre": genre,
                "count": count,
                "share": count / total,
            })
    # Explicit columns keep the schema stable even when no genres were found.
    return pd.DataFrame(rows, columns=GENRE_MAKEUP_COLUMNS)


def theme_statistics(
    themed: pd.DataFrame,
    min_support: int = 30,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute per-theme rating stats and the HIGH / LOW / baseline classification.

    A theme is HIGH when the lower bound of its ``confidence``-level
    normal-approximation interval for ``high_pct`` is above the global
    high-rate, LOW when the upper bound is below it, otherwise baseline.
    Themes with fewer than ``min_support`` movies are not classified at all
    (``"Excluded (small)"``). Rows come back in theme-id order via
    :func:`analysis.sorting.theme_id_key`.

    Preconditions:
        ``themed`` is a non-empty ``DataFrame`` with columns ``theme``,
        ``vote_average`` and ``high_rating`` (a non-empty frame is required
        because the global rate is undefined on zero movies);
        ``min_support`` is a non-negative ``int``; ``confidence`` is a
        fraction strictly between 0 and 1.
    """
    require_columns(themed, ["theme", "vote_average", "high_rating"], "themed")
    require_non_empty(themed, "themed")
    require_int(min_support, "min_support", minimum=0)
    require_fraction(confidence, "confidence", exclusive=True)

    global_rate = float(themed["high_rating"].mean())
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    rows: list[dict[str, Any]] = []
    for theme, group in themed.groupby("theme"):
        count = int(len(group))
        mean_rating = float(group["vote_average"].mean())
        high_pct = float(group["high_rating"].mean())
        lift = high_pct / global_rate if global_rate > 0 else float("nan")
        if count < min_support:
            classification = "Excluded (small)"
        else:
            se = np.sqrt(high_pct * (1.0 - high_pct) / count)
            if high_pct - z * se > global_rate:
                classification = "HIGH"
            elif high_pct + z * se < global_rate:
                classification = "LOW"
            else:
                classification = "baseline"
        rows.append({
            "theme": int(theme),
            "count": count,
            "mean_rating": mean_rating,
            "high_pct": high_pct,
            "lift": lift,
            "classification": classification,
            "global_rate": global_rate,
        })
    frame = pd.DataFrame(rows, columns=THEME_STATS_COLUMNS)
    return sort_frame(frame, theme_id_key).reset_index(drop=True)


def floor_sensitivity(
    metadata_clean: pd.DataFrame,
    floors: tuple[int, ...] = (25, 50, 100),
) -> pd.DataFrame:
    """Global high-rate at several vote floors, to test cutoff sensitivity.

    Preconditions:
        ``metadata_clean`` is a ``DataFrame`` with columns ``vote_average``
        and ``vote_count``; ``floors`` is a non-empty ``tuple`` of
        non-negative ``int`` values.
    """
    require_columns(metadata_clean, ["vote_average", "vote_count"], "metadata_clean")
    require_type(floors, tuple, "floors")
    require_non_empty(floors, "floors")
    require_all(
        floors,
        lambda item: isinstance(item, int) and not isinstance(item, bool) and item >= 0,
        "floors", "a non-negative int",
    )
    rows: list[dict[str, Any]] = []
    for floor in floors:
        sub = metadata_clean[metadata_clean["vote_count"] >= float(floor)]
        high_rate = float(sub["vote_average"].ge(HIGH_THRESHOLD).mean()) if len(sub) else float("nan")
        rows.append({"floor": int(floor), "count": len(sub), "high_rate": high_rate})
    return pd.DataFrame(rows)


def _top_genres_label(makeup: pd.DataFrame, theme: int, limit: int = 4) -> str:
    """Human-readable top genres for one theme, e.g. ``'Comedy(198), Action(72)'``.

    Genres are ordered with :func:`analysis.sorting.genre_count_key`: most
    mentions first, alphabetical among equal counts, so the label is stable
    across runs regardless of dict iteration order upstream.

    Preconditions:
        ``makeup`` is a ``DataFrame`` with columns ``theme``, ``genre`` and
        ``count`` (as built by :func:`theme_genre_makeup`); ``theme`` is an
        ``int``; ``limit`` is an ``int`` >= 1.
    """
    require_columns(makeup, ["theme", "genre", "count"], "makeup")
    require_int(theme, "theme")
    require_int(limit, "limit", minimum=1)
    subset = sort_frame(makeup[makeup["theme"] == theme], genre_count_key)
    return ", ".join(f"{row.genre}({row.count})" for row in subset.head(limit).itertuples())


def main() -> None:
    """Run stage 04 end-to-end and write theme labels/stats artifacts.

    Preconditions:
        stages 01-03 have been run, so ``themed_set.csv``,
        ``metadata_clean.csv``, ``vocab.json`` and the TF-IDF artifacts exist
        in ``analysis/output/``.
    """
    out = ensure_output_dir()
    for artifact in ("themed_set.csv", "metadata_clean.csv", "vocab.json"):
        require_existing_file(out / artifact, artifact)
    themed = pd.read_csv(out / "themed_set.csv")
    matrix, tfidf_ids = load_sparse(out)
    vocab_blob = load_json(out / "vocab.json")
    require_type(vocab_blob, dict, "vocab.json")
    vocab = vocab_blob["vocab"]
    require_type(vocab, list, "vocab.json['vocab']")

    top = theme_top_keywords(matrix, vocab, tfidf_ids, themed, top_n=8)
    makeup = theme_genre_makeup(themed)
    stats = theme_statistics(themed, min_support=30, confidence=0.95)

    metadata_clean = pd.read_csv(out / "metadata_clean.csv")
    sensitivity = floor_sensitivity(metadata_clean)

    label_rows: list[dict[str, Any]] = []
    for theme in sorted(top):
        for rank, keyword in enumerate(top[theme], start=1):
            label_rows.append({"theme": theme, "rank": rank, "keyword": keyword})
    pd.DataFrame(label_rows).to_csv(out / "theme_labels.csv", index=False)
    makeup.to_csv(out / "theme_genres.csv", index=False)
    sensitivity.to_csv(out / "floor_sensitivity.csv", index=False)

    stats["top_keywords"] = stats["theme"].map(lambda t: " | ".join(top.get(int(t), [])))
    stats["top_genres"] = stats["theme"].map(lambda t: _top_genres_label(makeup, int(t)))
    stats.to_csv(out / "theme_stats.csv", index=False)

    global_rate = float(stats["global_rate"].iloc[0])
    base_counts = stats["classification"].value_counts().to_dict()
    save_json({
        "global_rate": global_rate,
        "n_movies": len(themed),
        "n_high": int(themed["high_rating"].sum()),
        "n_themes": len(stats),
        "classification_counts": {str(k): int(v) for k, v in base_counts.items()},
        "floor_sensitivity": {
            str(int(r.floor)): {"count": int(r.count), "high_rate": float(r.high_rate)}
            for r in sensitivity.itertuples()
        },
    }, out / "global_stats.json")

    print(f"global high-rate: {global_rate:.4f} ({stats['count'].sum()} movies)")
    # Console "leaderboard": HIGH themes first, then baseline, then LOW, then
    # excluded; within each group the highest high-rate comes first.
    for row in sort_frame(stats, classification_rank_key).itertuples():
        print(f"  theme {row.theme:>2} [{row.classification:<14}] "
              f"n={row.count:>4} high={row.high_pct:.3f} "
              f"mean={row.mean_rating:.2f} | {row.top_keywords[:60]}")


if __name__ == "__main__":
    main()
