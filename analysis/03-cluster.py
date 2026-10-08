"""Stage 03: choose k and cluster movies into keyword themes."""

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

from analysis.common import ensure_output_dir, load_json, load_sparse, save_json

N_INIT: int = 10


def select_k(
    matrix: sparse.csr_matrix,
    k_min: int,
    k_max: int,
    seed: int,
) -> tuple[int, dict[int, float]]:
    """Pick k in [k_min, k_max] by silhouette; return (k, scores by k)."""
    n_rows = matrix.shape[0]
    hi = min(k_max, n_rows - 1)
    lo = max(2, k_min)
    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(matrix)
        scores[k] = float(silhouette_score(matrix, labels, metric="cosine"))
    best = max(scores, key=scores.get)
    return best, scores


def fit_themes(matrix: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Fit k-means and return integer theme labels per row."""
    labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(matrix)
    return labels.astype(np.int64)


def build_themed_set(
    working: pd.DataFrame,
    tfidf_ids: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach theme labels to movies; report sizes and too-small clusters."""
    label_map = dict(zip(tfidf_ids.astype(np.int64), labels.astype(np.int64)))
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
    themed = pd.DataFrame(rows)
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
    """Run stage 03 end-to-end and write themed set + cluster summary."""
    out = ensure_output_dir()
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
    """Read the TF-IDF artifacts written by stage 02."""
    return load_sparse(out)


if __name__ == "__main__":
    main()