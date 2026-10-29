"""Stage 03: choose k and cluster movies into keyword themes.

k-means is run once per candidate ``k``; the ``k`` with the best silhouette
score wins (ties go to the smaller ``k`` via a custom sort key, see
:func:`analysis.sorting.silhouette_key`). The winning labels are attached to
the working set and clusters below a minimum size are dropped so later
stages never draw conclusions from a handful of movies.

Every function documents its *preconditions* and enforces them with
:mod:`analysis.preconditions`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, load_sparse, save_json
from analysis.preconditions import (
    require,
    require_array_1d,
    require_columns,
    require_existing_dir,
    require_existing_file,
    require_int,
    require_ordered,
    require_same_length,
    require_sparse_matrix,
)
from analysis.sorting import silhouette_key

N_INIT: int = 10

#: Columns of the working set that :func:`build_themed_set` copies through.
WORKING_REQUIRED_COLUMNS: list[str] = [
    "id", "title", "genres", "keywords", "vote_average", "vote_count", "high_rating",
]

#: Columns of the themed set, in order (the working set plus ``theme``).
THEMED_COLUMNS: list[str] = WORKING_REQUIRED_COLUMNS + ["theme"]


def select_k(
    matrix: sparse.csr_matrix,
    k_min: int,
    k_max: int,
    seed: int,
) -> tuple[int, dict[int, float]]:
    """Pick ``k`` in ``[k_min, k_max]`` by silhouette; return ``(k, scores by k)``.

    The candidate range is clamped to what the silhouette score can evaluate
    (``2 <= k <= n_rows - 1``). The best ``k`` is chosen with
    :func:`analysis.sorting.silhouette_key`, so an exact tie between two
    scores resolves to the *smaller* ``k`` - the simpler model.

    Preconditions:
        ``matrix`` is a 2-d SciPy sparse matrix with at least 3 rows;
        ``k_min`` and ``k_max`` are ``int`` with ``1 <= k_min <= k_max``;
        ``seed`` is a non-negative ``int``; and the clamped range is
        non-empty, i.e. ``max(2, k_min) <= min(k_max, n_rows - 1)``.
    """
    require_sparse_matrix(matrix, "matrix")
    require_int(k_min, "k_min", minimum=1)
    require_int(k_max, "k_max", minimum=1)
    require_ordered(k_min, k_max, "k_min", "k_max")
    require_int(seed, "seed", minimum=0)
    n_rows = matrix.shape[0]
    require(n_rows >= 3, f"'matrix' needs at least 3 rows to score silhouettes, has {n_rows}")
    hi = min(k_max, n_rows - 1)
    lo = max(2, k_min)
    require(
        lo <= hi,
        f"no candidate k: {n_rows} rows allow k <= {n_rows - 1}, but k_min={k_min}",
    )
    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(matrix)
        scores[k] = float(silhouette_score(matrix, labels, metric="cosine"))
    best = max(scores, key=silhouette_key(scores))
    return best, scores


def fit_themes(matrix: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Fit k-means and return integer theme labels per row.

    Preconditions:
        ``matrix`` is a 2-d SciPy sparse matrix; ``k`` is an ``int`` with
        ``1 <= k <= n_rows``; ``seed`` is a non-negative ``int``.
    """
    require_sparse_matrix(matrix, "matrix")
    require_int(k, "k", minimum=1, maximum=int(matrix.shape[0]))
    require_int(seed, "seed", minimum=0)
    labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(matrix)
    return labels.astype(np.int64)


def build_themed_set(
    working: pd.DataFrame,
    tfidf_ids: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach theme labels to movies; report sizes and too-small clusters.

    ``tfidf_ids[i]`` is the movie id of matrix row ``i`` and ``labels[i]`` its
    cluster. Working-set movies without a matrix row (no keywords) are left
    out. Clusters with fewer than ``min_cluster_size`` members are dropped
    from the themed set but still listed in the summary.

    Preconditions:
        ``working`` is a ``DataFrame`` with every column in
        :data:`WORKING_REQUIRED_COLUMNS`; ``tfidf_ids`` and ``labels`` are
        1-d arrays (or lists) of equal length; ``min_cluster_size`` is a
        non-negative ``int``.
    """
    require_columns(working, WORKING_REQUIRED_COLUMNS, "working")
    require_array_1d(tfidf_ids, "tfidf_ids")
    require_array_1d(labels, "labels")
    require_same_length(tfidf_ids, labels, "tfidf_ids", "labels")
    require_int(min_cluster_size, "min_cluster_size", minimum=0)

    ids_array = np.asarray(tfidf_ids).astype(np.int64)
    labels_array = np.asarray(labels).astype(np.int64)
    label_map = dict(zip(ids_array, labels_array))
    rows: list[dict[str, Any]] = []
    for movie in working.itertuples(index=False):
        theme = label_map.get(int(movie.id))
        if theme is not None:
            rows.append({
                "id": int(movie.id),
                "title": movie.title,
                "genres": movie.genres,
                "keywords": movie.keywords,
                "vote_average": float(movie.vote_average),
                "vote_count": float(movie.vote_count),
                "high_rating": bool(movie.high_rating),
                "theme": int(theme),
            })
    # Passing explicit columns keeps the frame's schema even when no movie
    # matched, so downstream column preconditions still hold on an empty set.
    themed = pd.DataFrame(rows, columns=THEMED_COLUMNS)
    if themed.empty:
        sizes: dict[int, int] = {}
    else:
        sizes = {
            int(theme): int(count)
            for theme, count in themed["theme"].value_counts().sort_index().items()
        }
    excluded = [t for t, size in sizes.items() if size < min_cluster_size]
    if excluded:
        themed = themed[~themed["theme"].isin(excluded)]
    summary: dict[str, Any] = {
        "cluster_sizes": sizes,
        "min_cluster_size": min_cluster_size,
        "excluded_clusters": sorted(excluded),
        "n_dropped": sum(sizes[t] for t in excluded),
        "n_movies_themed": len(themed),
    }
    return themed.reset_index(drop=True), summary


def main() -> None:
    """Run stage 03 end-to-end and write themed set + cluster summary.

    Preconditions:
        stages 01 and 02 have been run, so ``working_set.csv``, ``tfidf.npz``
        and ``tfidf_ids.npy`` exist in ``analysis/output/``.
    """
    out = ensure_output_dir()
    require_existing_file(out / "working_set.csv", "working_set.csv")
    matrix, tfidf_ids = _load_matrix(out)
    working = pd.read_csv(out / "working_set.csv")

    best_k, scores = select_k(matrix, k_min=8, k_max=15, seed=42)
    labels = fit_themes(matrix, k=best_k, seed=42)
    themed, summary = build_themed_set(working, tfidf_ids, labels, min_cluster_size=30)

    themed.to_csv(out / "themed_set.csv", index=False)

    save_json({
        "k": best_k,
        "silhouette_scores": {str(k): round(v, 6) for k, v in scores.items()},
        "seed": 42,
        **summary,
    }, out / "cluster_summary.json")

    print(f"best k = {best_k} (silhouette={scores[best_k]:.4f})")
    print(f"cluster sizes: {summary['cluster_sizes']}")
    print(f"themed movies: {summary['n_movies_themed']} "
          f"(dropped {summary['n_dropped']} from small clusters "
          f"{summary['excluded_clusters']})")


def _load_matrix(out: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Read the TF-IDF artifacts written by stage 02.

    Preconditions:
        ``out`` is an existing directory containing the stage-02 artifacts
        (checked in detail by :func:`analysis.common.load_sparse`).
    """
    require_existing_dir(out, "out")
    return load_sparse(out)


if __name__ == "__main__":
    main()
