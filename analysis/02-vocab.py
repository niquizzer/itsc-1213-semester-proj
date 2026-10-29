"""Stage 02: curate the keyword vocabulary and build TF-IDF features.

Each movie's keyword list is lower-cased and de-duplicated, then turned into
a row of a TF-IDF matrix over a curated vocabulary (terms seen in at least
``min_df`` movies, at most ``max_df`` of them, capped at ``max_features``).
Movies with no keywords at all are excluded here and their ids recorded.

Every function documents its *preconditions* and enforces them with
:mod:`analysis.preconditions` before touching scikit-learn, whose own error
messages for bad arguments are far less direct.
"""

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
from analysis.preconditions import (
    require,
    require_all,
    require_existing_file,
    require_int,
    require_real,
    require_same_length,
    require_type,
)


def clean_keyword_list(keywords: list[str]) -> list[str]:
    """Lowercase, strip, and de-duplicate one movie's keyword list.

    First occurrence order is preserved so the cleaned list is deterministic
    for a given input (important for reproducible TF-IDF columns).

    Preconditions:
        ``keywords`` is a ``list`` whose elements are all ``str`` (empty
        strings are allowed and are dropped).
    """
    require_type(keywords, list, "keywords")
    require_all(keywords, lambda item: isinstance(item, str), "keywords", "a str")
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
    """Build a TF-IDF matrix, returning ``(matrix, vocab, kept ids, stats)``.

    ``keyword_lists[i]`` belongs to ``movie_ids[i]``. Movies whose list is
    empty are left out of the matrix (``stats["excluded_ids"]``); the rows of
    the returned matrix line up with ``kept_ids`` in order. Rows are
    L2-normalised by ``TfidfVectorizer`` so cosine distance is just a dot
    product for the clustering stage.

    Preconditions:
        ``keyword_lists`` and ``movie_ids`` are lists of equal length;
        every element of ``keyword_lists`` is itself a ``list``; every
        element of ``movie_ids`` is an ``int``; ``min_df`` is an ``int``
        >= 1; ``max_df`` is a fraction in ``(0, 1]``; ``max_features`` is an
        ``int`` >= 1; and at least one keyword list is non-empty (otherwise
        there is nothing to vectorise).
    """
    require_type(keyword_lists, list, "keyword_lists")
    require_type(movie_ids, list, "movie_ids")
    require_same_length(keyword_lists, movie_ids, "keyword_lists", "movie_ids")
    require_all(keyword_lists, lambda item: isinstance(item, list), "keyword_lists", "a list")
    require_all(
        movie_ids,
        lambda item: isinstance(item, int) and not isinstance(item, bool),
        "movie_ids", "an int",
    )
    require_int(min_df, "min_df", minimum=1)
    require_real(max_df, "max_df", minimum=0.0, maximum=1.0, exclusive_minimum=True)
    require_int(max_features, "max_features", minimum=1)

    kept_ids: list[int] = []
    kept_lists: list[list[str]] = []
    excluded_ids: list[int] = []
    for movie_id, keywords in zip(movie_ids, keyword_lists):
        if keywords:
            kept_ids.append(movie_id)
            kept_lists.append(keywords)
        else:
            excluded_ids.append(movie_id)
    require(kept_lists, "at least one movie must have a non-empty keyword list")

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
    """Run stage 02 end-to-end and write vocabulary + matrix artifacts.

    Preconditions:
        stage 01 has been run, so ``analysis/output/working_set.csv`` and
        ``analysis/output/keywords.json`` exist.
    """
    out = ensure_output_dir()
    require_existing_file(out / "working_set.csv", "working_set.csv")
    require_existing_file(out / "keywords.json", "keywords.json")
    working = pd.read_csv(out / "working_set.csv")
    keywords_by_id = load_json(out / "keywords.json")
    require_type(keywords_by_id, dict, "keywords.json")

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
