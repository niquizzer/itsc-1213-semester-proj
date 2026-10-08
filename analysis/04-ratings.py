"""Stage 04: label themes and compute per-theme rating outcomes."""

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

HIGH_THRESHOLD: float = 7.5


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


def theme_top_keywords(
    matrix: sparse.csr_matrix,
    vocab: list[str],
    tfidf_ids: np.ndarray,
    themed: pd.DataFrame,
    top_n: int,
) -> dict[int, list[str]]:
    """Return each theme's top_n characteristic keywords by mean TF-IDF."""
    row_of = {int(movie_id): row for row, movie_id in enumerate(tfidf_ids)}
    members: dict[int, list[int]] = {}
    for movie in themed.itertuples(index=False):
        row = row_of.get(int(movie.id))
        if row is not None:
            members.setdefault(int(movie.theme), []).append(row)
    result: dict[int, list[str]] = {}
    for theme, rows in sorted(members.items()):
        means = np.asarray(matrix[rows].mean(axis=0)).ravel()
        order = sorted(range(len(vocab)), key=lambda j: (-means[j], vocab[j]))
        result[theme] = [vocab[j] for j in order[:top_n]]
    return result


def theme_genre_makeup(themed: pd.DataFrame) -> pd.DataFrame:
    """Return per-theme genre value counts as a DataFrame."""
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
    return pd.DataFrame(rows)


def theme_statistics(
    themed: pd.DataFrame,
    min_support: int = 30,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute per-theme rating stats and HIGH/LOW classification."""
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
    return pd.DataFrame(rows).sort_values("theme").reset_index(drop=True)


def floor_sensitivity(
    metadata_clean: pd.DataFrame,
    floors: tuple[int, ...] = (25, 50, 100),
) -> pd.DataFrame:
    """Global high-rate at several vote floors, to test cutoff sensitivity."""
    rows: list[dict[str, Any]] = []
    for floor in floors:
        sub = metadata_clean[metadata_clean["vote_count"] >= float(floor)]
        high_rate = float(sub["vote_average"].ge(HIGH_THRESHOLD).mean()) if len(sub) else float("nan")
        rows.append({"floor": int(floor), "count": len(sub), "high_rate": high_rate})
    return pd.DataFrame(rows)


def _top_genres_label(makeup: pd.DataFrame, theme: int, limit: int = 4) -> str:
    """Human-readable top genres for one theme, e.g. 'Comedy(198), Action(72)'."""
    subset = makeup[makeup["theme"] == theme].sort_values("count", ascending=False)
    return ", ".join(f"{row.genre}({row.count})" for row in subset.head(limit).itertuples())


def main() -> None:
    """Run stage 04 end-to-end and write theme labels/stats artifacts."""
    out = ensure_output_dir()
    themed = pd.read_csv(out / "themed_set.csv")
    matrix, tfidf_ids = load_sparse(out)
    vocab = load_json(out / "vocab.json")["vocab"]
    assert isinstance(vocab, list)

    top = theme_top_keywords(matrix, vocab, tfidf_ids, themed, top_n=8)
    makeup = theme_genre_makeup(themed)
    stats = theme_statistics(themed, min_support=30, confidence=0.95)

    metadata_clean = pd.read_csv(out / "metadata_clean.csv")
    sensitivity = floor_sensitivity(metadata_clean)

    label_rows: list[dict[str, Any]] = []
    for theme, keywords in sorted(top.items()):
        for rank, keyword in enumerate(keywords, start=1):
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
    for row in stats.itertuples():
        print(f"  theme {row.theme:>2} [{row.classification:<14}] "
              f"n={row.count:>4} high={row.high_pct:.3f} "
              f"mean={row.mean_rating:.2f} | {row.top_keywords[:60]}")


if __name__ == "__main__":
    main()