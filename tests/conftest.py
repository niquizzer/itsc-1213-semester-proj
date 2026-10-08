"""Shared test fixtures built from tiny hand-written toy datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# The real movies_metadata.csv header, reproduced exactly for toy files.
METADATA_HEADER: list[str] = [
    "adult", "belongs_to_collection", "budget", "genres", "homepage", "id",
    "imdb_id", "original_language", "original_title", "overview", "popularity",
    "poster_path", "production_companies", "production_countries",
    "release_date", "revenue", "runtime", "spoken_languages", "status",
    "tagline", "title", "video", "vote_average", "vote_count",
]


def _metadata_row(row: dict[str, str]) -> list[str]:
    """Expand a sparse dict of real columns into a full header-width row."""
    return [row.get(key, "") for key in METADATA_HEADER]


def write_toy_metadata(path: Path) -> Path:
    """Write a toy movies_metadata.csv exercising every loader edge case."""
    rows: list[list[str]] = [
        _metadata_row({
            "id": "862", "title": "Toy Story",
            "genres": "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]",
            "vote_average": "7.7", "vote_count": "5415",
        }),
        _metadata_row({
            "id": "8844", "title": "Jumanji",
            "genres": "[{'id': 12, 'name': 'Adventure'}]",
            "vote_average": "6.9", "vote_count": "2413",
        }),
        _metadata_row({
            "id": "100", "title": "Boundary High",
            "genres": "[{'id': 18, 'name': 'Drama'}]",
            "vote_average": "7.5", "vote_count": "50",
        }),
        _metadata_row({
            "id": "101", "title": "Boundary Low",
            "genres": "[{'id': 27, 'name': 'Horror'}]",
            "vote_average": "7.49", "vote_count": "80",
        }),
        _metadata_row({
            "id": "102", "title": "Low Vote",
            "genres": "[{'id': 27, 'name': 'Horror'}]",
            "vote_average": "8.0", "vote_count": "49",
        }),
        _metadata_row({
            "id": "103", "title": "At Floor",
            "genres": "[{'id': 9648, 'name': 'Mystery'}]",
            "vote_average": "6.0", "vote_count": "50",
        }),
        _metadata_row({
            "id": "104", "title": "No Genres, No Keywords",
            "genres": "", "vote_average": "5.5", "vote_count": "90",
        }),
        _metadata_row({
            "id": "105", "title": "Garbage Keywords",
            "genres": "[{'id': 35, 'name': 'Comedy'}]",
            "vote_average": "7.2", "vote_count": "200",
        }),
        _metadata_row({
            "id": "200", "title": "Bad Vote Average",
            "genres": "[{'id': 35, 'name': 'Comedy'}]",
            "vote_average": "Toy Story", "vote_count": "100",
        }),
        _metadata_row({
            "id": "9abc", "title": "Bad Id",
            "genres": "[]", "vote_average": "5.0", "vote_count": "100",
        }),
        # Duplicate of id 862 (loader keeps the first occurrence).
        _metadata_row({
            "id": "862", "title": "Toy Story Duplicate",
            "genres": "[{'id': 16, 'name': 'Animation'}]",
            "vote_average": "6.0", "vote_count": "10",
        }),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_HEADER)
        writer.writerows(rows)
    return path


def write_toy_keywords(path: Path) -> Path:
    """Write a toy keywords.csv, including a malformed cell and a spare id."""
    rows: list[tuple[str, str]] = [
        ("862", "[{'id': 931, 'name': 'Jealousy'}, {'id': 4290, 'name': 'toy'}, "
                "{'id': 5202, 'name': 'Boy'}, {'id': 6054, 'name': 'friendship'}, "
                "{'id': 9713, 'name': 'friends'}, {'id': 9823, 'name': 'Toy'}, "
                "{'id': 9824, 'name': 'jealousy'}]"),
        # Field with a quoted phrase (exercises CSV escaping + literal_eval).
        ("8844", "[{'id': 15101, 'name': \"based on children's book\"}, "
                 "{'id': 10090, 'name': 'board game'}, "
                 "{'id': 10941, 'name': 'disappearance'}, "
                 "{'id': 10090, 'name': 'board game'}]"),
        ("100", "[{'id': 1, 'name': 'investigation'}, {'id': 2, 'name': 'murder'}]"),
        ("101", "[{'id': 3, 'name': 'murder'}, {'id': 4, 'name': 'gore'}, "
                "{'id': 5, 'name': 'haunted house'}]"),
        ("102", "[{'id': 6, 'name': 'gore'}, {'id': 7, 'name': 'franchise'}]"),
        ("103", "[{'id': 8, 'name': 'whodunit'}, {'id': 9, 'name': 'detective'}]"),
        # Duplicate keywords row for id 103 (loader keeps the first).
        ("103", "[{'id': 8, 'name': 'whodunit'}]"),
        # Parses to an empty list (movie is excluded at vocabulary stage).
        ("105", "not a valid python literal"),
        # Keywords row whose movie is missing from metadata (ignored at join).
        ("999", "[{'id': 10, 'name': 'orphan'}]"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "keywords"])
        writer.writerows(rows)
    return path


def write_toy_links(path: Path) -> Path:
    """Write a toy links.csv bridging MovieLens ids to TMDB ids."""
    rows: list[tuple[str, str, str]] = [
        ("1", "0114709", "862"),
        ("2", "0113497", "8844"),
        ("3", "0000100", "100"),
        ("4", "0000103", "103"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["movieId", "imdbId", "tmdbId"])
        writer.writerows(rows)
    return path


def write_toy_ratings(path: Path) -> Path:
    """Write a toy ratings_small.csv; movie id 101 has no ratings at all."""
    rows: list[tuple[str, str, str, str]] = [
        ("1", "1", "4.0", "1425941529"),
        ("1", "2", "4.5", "1425942435"),
        ("2", "2", "4.5", "1425942435"),
        ("3", "1", "5.0", "1425942435"),
        ("4", "3", "3.0", "1425942435"),
        ("5", "3", "3.0", "1425942435"),
        ("6", "4", "2.5", "1425942435"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerows(rows)
    return path


@pytest.fixture
def toy_metadata_csv(tmp_path: Path) -> Path:
    """Path to a toy movies_metadata.csv exercising loader edge cases."""
    return write_toy_metadata(tmp_path / "movies_metadata.csv")


@pytest.fixture
def toy_keywords_csv(tmp_path: Path) -> Path:
    """Path to a toy keywords.csv covering valid, quoted, and malformed cells."""
    return write_toy_keywords(tmp_path / "keywords.csv")


@pytest.fixture
def toy_links_csv(tmp_path: Path) -> Path:
    """Path to a toy links.csv."""
    return write_toy_links(tmp_path / "links.csv")


@pytest.fixture
def toy_ratings_csv(tmp_path: Path) -> Path:
    """Path to a toy ratings_small.csv."""
    return write_toy_ratings(tmp_path / "ratings.csv")


@pytest.fixture
def toy_keyword_lists() -> list[list[str]]:
    """Eight hand-built keyword lists with controlled document frequencies.

    Term document counts: alpha/beta/zeta/eta appear in 3 docs, omega/sigma in
    2, everything else in 1.
    """
    return [
        ["alpha", "beta", "gamma"],
        ["alpha", "beta", "delta"],
        ["alpha", "beta", "epsilon"],
        ["zeta", "eta", "theta"],
        ["zeta", "eta", "iota"],
        ["zeta", "eta", "kappa"],
        ["omega", "sigma", "tau"],
        ["omega", "sigma"],
    ]


@pytest.fixture
def toy_themed_df() -> pd.DataFrame:
    """A small themed set for rating-statistic tests (24 movies, 4 themes)."""
    dicts: list[dict[str, Any]] = []
    # theme 0: 8/8 high, theme 1: 0/8, theme 2: 3/8 (matches a 37.5% global),
    # theme 3: only 2 movies (below the test's minimum support of 8).
    high_counts = {0: 8, 1: 0, 2: 3, 3: 1}
    for theme, n_high in high_counts.items():
        n = 8 if theme != 3 else 2
        for i in range(n):
            dicts.append({
                "id": 1000 + theme * 100 + i,
                "title": f"movie {theme}-{i}",
                "genres": ["Drama"],
                "vote_average": 8.0 if i < n_high else 5.0,
                "vote_count": float(100 + i),
                "high_rating": i < n_high,
                "theme": theme,
            })
    return pd.DataFrame(dicts)