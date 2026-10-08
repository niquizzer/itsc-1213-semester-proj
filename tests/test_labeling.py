"""Tests for stage 04-ratings theme labeling: top keywords + genre make-up."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis import common

ratings_mod = common.load_script("04-ratings")

LISTS = [
    ["alpha", "u1"],
    ["alpha", "u2"],
    ["alpha", "u3"],
    ["beta", "v1"],
    ["beta", "v2"],
    ["beta", "v3"],
]
IDS = [101, 102, 103, 201, 202, 203]   # theme 0: 101-103, theme 1: 201-203
THEMES = [0, 0, 0, 1, 1, 1]


def labeled_movies() -> tuple[sparse.csr_matrix, list[str], list[int]]:
    """Six movies across two themes; each theme has one shared marker keyword."""
    vectorizer = TfidfVectorizer(
        lowercase=False, preprocessor=lambda d: d, tokenizer=lambda d: d,
        token_pattern=None,
    )
    matrix = vectorizer.fit_transform(LISTS)
    vocab = list(vectorizer.get_feature_names_out())
    return matrix, vocab, IDS


def themed_frame(genres: list[list[str]] | None = None) -> pd.DataFrame:
    """A themed DataFrame; genres default to a shared marker per theme."""
    if genres is None:
        genres = [["Animation", "Comedy"], ["Comedy"], [], ["Drama"], ["Drama"], []]
    return pd.DataFrame({
        "id": IDS,
        "title": [f"m{i}" for i in IDS],
        "genres": genres,
        "keywords": LISTS,
        "vote_average": [8.0, 7.8, 7.6, 4.0, 4.2, 4.4],
        "vote_count": [100.0] * 6,
        "high_rating": [True, True, True, False, False, False],
        "theme": THEMES,
    })


def reference_top_keywords(
    matrix: sparse.csr_matrix, vocab: list[str], themed: pd.DataFrame, top_n: int,
) -> dict[int, list[str]]:
    """Reference rule: per theme, rank terms by mean TF-IDF, ties alphabetically."""
    row_of = {int(movie_id): row for row, movie_id in enumerate(IDS)}
    members: dict[int, list[int]] = {0: [], 1: []}
    for movie in themed.itertuples():
        members[int(movie.theme)].append(row_of[int(movie.id)])
    result: dict[int, list[str]] = {}
    for theme, rows in members.items():
        means = np.asarray(matrix[rows].mean(axis=0)).ravel()
        order = sorted(range(len(vocab)), key=lambda j: (-means[j], vocab[j]))
        result[theme] = [vocab[j] for j in order[:top_n]]
    return result


def test_top_keywords_match_mean_tfidf_rule() -> None:
    """Each theme's top keywords are its marker first, then alphabetically."""
    matrix, vocab, _ = labeled_movies()
    themed = themed_frame([["Animation"], ["Animation"], ["Animation"],
                           ["Drama"], ["Drama"], ["Drama"]])
    top = ratings_mod.theme_top_keywords(matrix, vocab, IDS, themed, top_n=3)
    assert set(top) == {0, 1}
    assert top[0][0] == "alpha"
    assert top[1][0] == "beta"
    assert top[0] == reference_top_keywords(matrix, vocab, themed, top_n=3)[0]
    assert top[1] == reference_top_keywords(matrix, vocab, themed, top_n=3)[1]


def test_top_keywords_top_n_limit() -> None:
    """top_n caps the number of returned keywords per theme."""
    matrix, vocab, _ = labeled_movies()
    themed = themed_frame()
    top = ratings_mod.theme_top_keywords(matrix, vocab, IDS, themed, top_n=1)
    assert all(len(keywords) == 1 for keywords in top.values())
    assert set(top.keys()) == {0, 1}


def test_genre_makeup_counts_and_shares() -> None:
    """Genre make-up counts mentions per theme and shares within theme."""
    themed = themed_frame([["Animation", "Comedy"], ["Comedy"], [],
                           ["Drama"], ["Drama"], []])
    makeup = ratings_mod.theme_genre_makeup(themed)
    theme0 = makeup[makeup["theme"] == 0].set_index("genre")
    assert theme0.loc["Animation", "count"] == 1
    assert theme0.loc["Comedy", "count"] == 2
    assert abs(theme0["share"].sum() - 1.0) < 1e-9
    theme1 = makeup[makeup["theme"] == 1].set_index("genre")
    assert theme1.loc["Drama", "count"] == 2
    assert abs(theme1.loc["Drama", "share"] - 1.0) < 1e-9


def test_genre_makeup_handles_missing_genres() -> None:
    """Movies without genres contribute nothing and don't crash the pass."""
    themed = themed_frame([[], [], [], [], [], []])
    makeup = ratings_mod.theme_genre_makeup(themed)
    assert makeup.empty


def test_genre_makeup_accepts_json_string_cells() -> None:
    """Themed sets persisted to CSV hold genres as JSON strings; parse them."""
    themed = themed_frame([["Drama"], ["Drama"], [], ["Drama"], [], []])
    json_frame = themed.copy()
    json_frame["genres"] = themed["genres"].map(__import__("json").dumps)
    makeup = ratings_mod.theme_genre_makeup(json_frame)
    counts = makeup[makeup["theme"] == 1].set_index("genre")["count"]
    assert counts.loc["Drama"] == 1