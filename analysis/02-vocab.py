"""Stage 02: curate the keyword vocabulary and build TF-IDF features."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, load_json, save_json, save_sparse


def clean_keyword_list(keywords: list[str]) -> list[str]:
    """Lowercase, strip, and de-duplicate one movie's keyword list."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for keyword in keywords:
        normalized = keyword.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def build_tfidf(
    keyword_lists: list[list[str]],
    movie_ids: list[int],
    min_df: int = 3,
    max_df: float = 0.4,
    max_features: int = 1000,
) -> tuple[sparse.csr_matrix, list[str], list[int], dict[str, Any]]:
    """Build a TF-IDF matrix, returning (matrix, vocab, kept ids, stats)."""
    kept_ids: list[int] = []
    kept_lists: list[list[str]] = []
    excluded_ids: list[int] = []
    for movie_id, keywords in zip(movie_ids, keyword_lists):
        if keywords:
            kept_ids.append(movie_id)
            kept_lists.append(keywords)
        else:
            excluded_ids.append(movie_id)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        preprocessor=lambda doc: doc,
        tokenizer=lambda doc: doc,
        token_pattern=None,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(kept_lists)
    vocab = list(vectorizer.get_feature_names_out())
    stats: dict[str, Any] = {
        "excluded_ids": excluded_ids,
        "n_excluded": len(excluded_ids),
        "vocab_size": len(vocab),
        "n_docs": len(kept_ids),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }
    return matrix, vocab, kept_ids, stats


def main() -> None:
    """Run stage 02 end-to-end and write vocabulary + matrix artifacts."""
    out = ensure_output_dir()
    working = pd.read_csv(out / "working_set.csv")
    keywords_by_id = load_json(out / "keywords.json")
    assert isinstance(keywords_by_id, dict)

    ids = [int(x) for x in working["id"]]
    keyword_lists = [
        clean_keyword_list(keywords_by_id.get(str(movie_id), [])) for movie_id in ids
    ]
    matrix, vocab, kept_ids, stats = build_tfidf(keyword_lists, ids)

    save_sparse(matrix, np.array(kept_ids, dtype=np.int64), out)
    save_json({"vocab": vocab, **stats}, out / "vocab.json")

    print(f"movies with keywords: {stats['n_docs']} "
          f"(excluded: {stats['n_excluded']})")
    print(f"vocabulary size: {stats['vocab_size']} "
          f"(min_df={stats['min_df']}, max_df={stats['max_df']}, "
          f"cap={stats['max_features']})")
    print(f"TF-IDF matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()