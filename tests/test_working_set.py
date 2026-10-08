"""Tests for stage 01-load: working-set construction (tests-first)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis import common

load = common.load_script("01-load")


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scratch output dir shared by the loader tests."""
    return tmp_path_factory.mktemp("out")


def ids(frame: pd.DataFrame) -> set[int]:
    """Return the id set of a working-set or metadata frame."""
    return set(int(x) for x in frame["id"])


def test_parse_dict_list_handles_malformed_and_quoted() -> None:
    """Unparsable cells become [], quoted phrases are preserved."""
    parsed = load.parse_dict_list("[{'id': 1, 'name': 'board game'}]")
    assert parsed == [{"id": 1, "name": "board game"}]
    assert load.parse_dict_list("not a literal") == []
    assert load.parse_dict_list("") == []
    assert load.parse_dict_list(None) == []
    quoted = load.parse_dict_list("[{'id': 15101, 'name': \"based on children's book\"}]")
    assert quoted[0]["name"] == "based on children's book"


def test_parse_genre_names_extracts_names() -> None:
    """Genre cells become plain name lists; empty cells become []."""
    raw = "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]"
    assert load.parse_genre_names(raw) == ["Animation", "Comedy"]
    assert load.parse_genre_names("") == []


def test_load_metadata_drops_malformed_rows(toy_metadata_csv: Path, pipeline_output: Path) -> None:
    """Shifted/garbage rows are dropped and each drop reason is counted."""
    metadata, stats = load.load_metadata(toy_metadata_csv, pipeline_output)
    assert ids(metadata) == {862, 8844, 100, 101, 102, 103, 104, 105}
    assert stats["dropped_bad_id"] == 1      # id "9abc"
    assert stats["dropped_bad_va"] == 1      # vote_average "Toy Story"
    assert stats["dropped_duplicate"] == 1   # second id 862 row
    assert stats["kept"] == 8


def test_load_metadata_parses_genres_and_numbers(
    toy_metadata_csv: Path, pipeline_output: Path
) -> None:
    """Genres become lists, ratings and votes become floats."""
    metadata, _ = load.load_metadata(toy_metadata_csv, pipeline_output)
    row = metadata.set_index("id").loc[862]
    assert row["genres"] == ["Animation", "Comedy"]
    assert float(row["vote_average"]) == 7.7
    assert float(row["vote_count"]) == 5415.0
    assert metadata.set_index("id").loc[104]["genres"] == []


def test_load_keywords_frames(toy_keywords_csv: Path, pipeline_output: Path) -> None:
    """Keyword lists parse; malformed cells become []; bad ids are dropped."""
    keywords, stats = load.load_keywords(toy_keywords_csv, pipeline_output)
    by_id = dict(zip(keywords["id"], keywords["keywords"]))
    assert set(by_id) == {862, 8844, 100, 101, 102, 103, 105, 999}
    assert stats["dropped_duplicate"] == 1  # duplicate row for id 103
    assert by_id[105] == []
    # Quoted phrase survives parsing, and duplicates are kept for now.
    assert "based on children's book" in by_id[8844]
    assert by_id[862].count("Toy") == 1 and by_id[862].count("toy") == 1


def test_build_working_set_join_floor_and_high_label(
    toy_metadata_csv: Path, toy_keywords_csv: Path, pipeline_output: Path
) -> None:
    """Join on id, vote floor 50, and the >= 7.5 high label all apply."""
    metadata, _ = load.load_metadata(toy_metadata_csv, pipeline_output)
    keywords, _ = load.load_keywords(toy_keywords_csv, pipeline_output)
    working, stats = load.build_working_set(metadata, keywords, min_votes=50)

    assert ids(working) == {862, 8844, 100, 101, 103, 105}
    assert stats["below_floor"] == 1        # id 102 (49 votes)
    assert stats["missing_keywords"] == 1   # id 104
    assert stats["missing_metadata"] == 1   # keyword-only id 999
    assert stats["kept"] == 6

    high = working.set_index("id")["high_rating"]
    assert bool(high.loc[862])              # 7.7
    assert bool(high.loc[100])              # 7.5 exactly -> high
    assert not bool(high.loc[101])          # 7.49 -> not high
    assert not bool(high.loc[103])          # 6.0
    assert not bool(high.loc[105])          # 7.2


def test_vote_floor_boundary(
    toy_metadata_csv: Path, toy_keywords_csv: Path, pipeline_output: Path
) -> None:
    """49 votes is excluded at floor 50; 50 votes is included."""
    metadata, _ = load.load_metadata(toy_metadata_csv, pipeline_output)
    keywords, _ = load.load_keywords(toy_keywords_csv, pipeline_output)
    working, _ = load.build_working_set(metadata, keywords, min_votes=50)
    assert 102 not in ids(working)
    assert 103 in ids(working)
    at_49, _ = load.build_working_set(metadata, keywords, min_votes=49)
    assert 102 in ids(at_49)


def test_high_rating_boundary(toy_metadata_csv: Path, pipeline_output: Path) -> None:
    """vote_average exactly 7.5 is high; 7.49 is not."""
    metadata, _ = load.load_metadata(toy_metadata_csv, pipeline_output)
    labels = metadata.set_index("id")["vote_average"].astype(float)
    assert float(labels.loc[100]) >= 7.5
    assert not float(labels.loc[101]) >= 7.5