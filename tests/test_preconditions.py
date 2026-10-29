"""Tests for analysis.preconditions and for the contracts the stages enforce."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from analysis import common
from analysis import preconditions as pre

vocab_mod = common.load_script("02-vocab")
cluster_mod = common.load_script("03-cluster")
ratings_mod = common.load_script("04-ratings")
validate_mod = common.load_script("05-validate")
report_mod = common.load_script("06-report")


# --- exception hierarchy ----------------------------------------------------


def test_precondition_error_is_a_value_error() -> None:
    """Existing ``except ValueError`` handlers keep working."""
    assert issubclass(pre.PreconditionError, ValueError)


def test_type_precondition_error_is_both_kinds() -> None:
    """A type violation can be caught as TypeError, ValueError or PreconditionError."""
    assert issubclass(pre.TypePreconditionError, pre.PreconditionError)
    assert issubclass(pre.TypePreconditionError, TypeError)
    with pytest.raises(TypeError):
        pre.require_type(1, str, "x")
    with pytest.raises(ValueError):
        pre.require_type(1, str, "x")


# --- helpers: pass / fail / message -----------------------------------------


def test_require_passes_truthy_and_fails_falsy() -> None:
    """require() is the general escape hatch; the message names the problem."""
    pre.require(True, "fine")
    with pytest.raises(pre.PreconditionError, match="custom problem"):
        pre.require(False, "custom problem")
    with pytest.raises(pre.TypePreconditionError):
        pre.require(True, "")


def test_require_type_message_names_argument_and_value() -> None:
    """Messages carry the argument name, expected type and actual value."""
    with pytest.raises(pre.TypePreconditionError, match=r"'k' must be int, got str 'two'"):
        pre.require_type("two", int, "k")
    pre.require_type(Path("x"), (Path, str), "path")
    with pytest.raises(pre.TypePreconditionError, match=r"Path \| str"):
        pre.require_type(3, (Path, str), "path")


def test_require_int_rejects_bool_and_enforces_bounds() -> None:
    """bool is not an int here; bounds are inclusive; NumPy ints are accepted."""
    pre.require_int(5, "n", minimum=5, maximum=5)
    pre.require_int(np.int64(3), "n", minimum=0)
    with pytest.raises(pre.TypePreconditionError):
        pre.require_int(True, "n")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_int(2.0, "n")
    with pytest.raises(pre.PreconditionError, match=">= 1"):
        pre.require_int(0, "n", minimum=1)
    with pytest.raises(pre.PreconditionError, match="<= 10"):
        pre.require_int(11, "n", maximum=10)
    with pytest.raises(pre.PreconditionError, match="exceeds"):
        pre.require_int(1, "n", minimum=5, maximum=1)


def test_require_real_bounds_exclusivity_and_nan() -> None:
    """Inclusive vs exclusive bounds and NaN handling behave as documented."""
    pre.require_real(0.0, "x", minimum=0.0)
    pre.require_real(np.float64(0.5), "x", minimum=0.0, maximum=1.0)
    pre.require_real(math.inf, "x", minimum=0.0, exclusive_minimum=True)
    with pytest.raises(pre.PreconditionError, match="> 0.0"):
        pre.require_real(0.0, "x", minimum=0.0, exclusive_minimum=True)
    with pytest.raises(pre.PreconditionError, match="< 1.0"):
        pre.require_real(1.0, "x", maximum=1.0, exclusive_maximum=True)
    with pytest.raises(pre.PreconditionError, match="NaN"):
        pre.require_real(math.nan, "x")
    pre.require_real(math.nan, "x", allow_nan=True)
    with pytest.raises(pre.TypePreconditionError):
        pre.require_real("0.5", "x")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_real(False, "x")


def test_require_fraction() -> None:
    """[0, 1] by default, (0, 1) when exclusive."""
    pre.require_fraction(0.0, "p")
    pre.require_fraction(1.0, "p")
    with pytest.raises(pre.PreconditionError):
        pre.require_fraction(1.0, "p", exclusive=True)
    with pytest.raises(pre.PreconditionError):
        pre.require_fraction(1.5, "p")


def test_require_ordered_non_empty_same_length() -> None:
    """The small structural helpers pass and fail as expected."""
    pre.require_ordered(1, 1, "lo", "hi")
    with pytest.raises(pre.PreconditionError, match="must be <="):
        pre.require_ordered(2, 1, "lo", "hi")
    pre.require_non_empty([0], "items")
    with pytest.raises(pre.PreconditionError, match="must not be empty"):
        pre.require_non_empty([], "items")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_non_empty(5, "items")
    pre.require_same_length([1, 2], (3, 4), "a", "b")
    with pytest.raises(pre.PreconditionError, match="same length"):
        pre.require_same_length([1], [1, 2], "a", "b")


def test_require_all_reports_first_failure_position() -> None:
    """The failing element and its position appear in the message."""
    pre.require_all([1, 2, 3], lambda v: v > 0, "xs", "positive")
    with pytest.raises(pre.PreconditionError, match="element 1 is int 0"):
        pre.require_all([1, 0, 3], lambda v: v > 0, "xs", "positive")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_all([1], "not callable", "xs", "positive")


def test_require_columns_lists_every_missing_column() -> None:
    """All missing columns are reported together; empty frames with columns pass."""
    frame = pd.DataFrame({"theme": [], "count": []})
    pre.require_columns(frame, ["theme", "count"], "f")
    with pytest.raises(pre.PreconditionError, match=r"\['high_pct', 'flag'\]"):
        pre.require_columns(frame, ["theme", "high_pct", "flag"], "f")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_columns({"theme": []}, ["theme"], "f")


def test_require_keys() -> None:
    """Mappings must carry every requested key."""
    pre.require_keys({"a": 1, "b": 2}, ["a"], "m")
    with pytest.raises(pre.PreconditionError, match=r"\['c'\]"):
        pre.require_keys({"a": 1}, ["a", "c"], "m")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_keys([("a", 1)], ["a"], "m")


def test_require_existing_file_and_dir(tmp_path: Path) -> None:
    """Files and directories are told apart; str paths are accepted."""
    file_path = tmp_path / "f.txt"
    file_path.write_text("x", encoding="utf-8")
    pre.require_existing_file(file_path, "p")
    pre.require_existing_file(str(file_path), "p")
    pre.require_existing_dir(tmp_path, "d")
    with pytest.raises(pre.PreconditionError, match="existing file"):
        pre.require_existing_file(tmp_path, "p")
    with pytest.raises(pre.PreconditionError, match="existing directory"):
        pre.require_existing_dir(file_path, "d")
    with pytest.raises(pre.PreconditionError):
        pre.require_existing_file(tmp_path / "missing.csv", "p")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_existing_file(42, "p")


def test_require_sparse_matrix_and_array_1d() -> None:
    """Sparse-matrix and 1-d array checks accept the right shapes only."""
    pre.require_sparse_matrix(sparse.csr_matrix((2, 2)), "m")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_sparse_matrix(np.zeros((2, 2)), "m")
    pre.require_array_1d(np.arange(3), "a")
    pre.require_array_1d([1, 2, 3], "a")
    pre.require_array_1d(pd.Series([1]), "a")
    with pytest.raises(pre.PreconditionError, match="1-dimensional"):
        pre.require_array_1d(np.zeros((2, 2)), "a")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_array_1d(7, "a")


def test_require_callable() -> None:
    """Functions and lambdas pass; plain values do not."""
    pre.require_callable(len, "fn")
    pre.require_callable(lambda: None, "fn")
    with pytest.raises(pre.TypePreconditionError):
        pre.require_callable("len", "fn")


# --- contracts enforced by the pipeline stages ------------------------------


def test_common_helpers_enforce_their_contracts(tmp_path: Path) -> None:
    """load_script / load_json / save_sparse / load_sparse reject bad input early."""
    with pytest.raises(pre.PreconditionError):
        common.load_script("99-nope")
    with pytest.raises(pre.PreconditionError):
        common.load_json(tmp_path / "missing.json")
    with pytest.raises(pre.PreconditionError):
        common.load_sparse(tmp_path)  # stage 02 never ran here
    matrix = sparse.csr_matrix(np.eye(3))
    with pytest.raises(pre.PreconditionError, match="3 rows"):
        common.save_sparse(matrix, np.arange(2), tmp_path)
    common.save_sparse(matrix, np.arange(3), tmp_path)
    loaded, ids = common.load_sparse(tmp_path)
    assert loaded.shape == (3, 3) and ids.tolist() == [0, 1, 2]


def test_build_tfidf_contracts(toy_keyword_lists: list[list[str]]) -> None:
    """Parallel-list, range and non-empty-corpus preconditions are enforced."""
    ids = list(range(len(toy_keyword_lists)))
    with pytest.raises(pre.PreconditionError, match="same length"):
        vocab_mod.build_tfidf(toy_keyword_lists, ids[:-1])
    with pytest.raises(pre.PreconditionError, match="min_df"):
        vocab_mod.build_tfidf(toy_keyword_lists, ids, min_df=0)
    with pytest.raises(pre.PreconditionError, match="max_df"):
        vocab_mod.build_tfidf(toy_keyword_lists, ids, max_df=1.5)
    with pytest.raises(pre.PreconditionError, match="max_features"):
        vocab_mod.build_tfidf(toy_keyword_lists, ids, max_features=0)
    with pytest.raises(pre.PreconditionError, match="non-empty keyword list"):
        vocab_mod.build_tfidf([[], []], [1, 2])
    with pytest.raises(pre.PreconditionError):
        vocab_mod.clean_keyword_list(["ok", 3])


def test_cluster_contracts() -> None:
    """k ranges, seeds and row counts are validated before k-means runs."""
    matrix = sparse.csr_matrix(np.eye(4))
    with pytest.raises(pre.PreconditionError, match="k_min"):
        cluster_mod.select_k(matrix, k_min=3, k_max=2, seed=42)
    with pytest.raises(pre.PreconditionError, match="seed"):
        cluster_mod.select_k(matrix, k_min=2, k_max=3, seed=-1)
    with pytest.raises(pre.PreconditionError, match="no candidate k"):
        cluster_mod.select_k(matrix, k_min=4, k_max=9, seed=42)
    with pytest.raises(pre.PreconditionError, match="at least 3 rows"):
        cluster_mod.select_k(sparse.csr_matrix(np.eye(2)), k_min=2, k_max=2, seed=42)
    with pytest.raises(pre.PreconditionError, match="'k' must be <= 4"):
        cluster_mod.fit_themes(matrix, k=5, seed=42)
    with pytest.raises(pre.TypePreconditionError):
        cluster_mod.fit_themes(np.eye(4), k=2, seed=42)
    working = pd.DataFrame({"id": [1]})
    with pytest.raises(pre.PreconditionError, match="missing required column"):
        cluster_mod.build_themed_set(working, np.array([1]), np.array([0]))


def test_ratings_contracts(toy_themed_df: pd.DataFrame) -> None:
    """Confidence, support and column contracts of stage 04."""
    with pytest.raises(pre.PreconditionError, match="confidence"):
        ratings_mod.theme_statistics(toy_themed_df, confidence=1.0)
    with pytest.raises(pre.PreconditionError, match="confidence"):
        ratings_mod.theme_statistics(toy_themed_df, confidence=0.0)
    with pytest.raises(pre.PreconditionError, match="min_support"):
        ratings_mod.theme_statistics(toy_themed_df, min_support=-1)
    with pytest.raises(pre.PreconditionError, match="must not be empty"):
        ratings_mod.theme_statistics(toy_themed_df.iloc[0:0])
    with pytest.raises(pre.PreconditionError, match="floors"):
        ratings_mod.floor_sensitivity(pd.DataFrame({"vote_average": [], "vote_count": []}), floors=())
    with pytest.raises(pre.PreconditionError, match="top_n"):
        ratings_mod.theme_top_keywords(
            sparse.csr_matrix(np.eye(2)), ["a", "b"], [1, 2], toy_themed_df, top_n=0,
        )
    with pytest.raises(pre.PreconditionError, match="columns"):
        ratings_mod.theme_top_keywords(
            sparse.csr_matrix(np.eye(2)), ["a"], [1, 2], toy_themed_df, top_n=1,
        )
    with pytest.raises(pre.PreconditionError, match="limit"):
        ratings_mod._top_genres_label(ratings_mod.theme_genre_makeup(toy_themed_df), 0, limit=0)


def test_validate_contracts(tmp_path: Path) -> None:
    """Genre coverage, rate ranges, bootstrap sizes and file paths are checked."""
    frame = pd.DataFrame({
        "id": [1, 2], "genres": [["Horror"], ["Drama"]],
        "high_rating": [True, False], "theme": [0, 0],
    })
    with pytest.raises(pre.PreconditionError, match=r"no entry for genre\(s\) \['Drama'\]"):
        validate_mod.expected_high_rate(frame, {"Horror": 0.3})
    with pytest.raises(pre.PreconditionError, match=r"rate in \[0, 1\]"):
        validate_mod.expected_high_rate(frame, {"Horror": 0.3, "Drama": 1.3})
    with pytest.raises(pre.PreconditionError, match="must not be empty"):
        validate_mod.bootstrap_mean_ci(np.array([]), n_boot=10, seed=1)
    with pytest.raises(pre.PreconditionError, match="n_boot"):
        validate_mod.bootstrap_mean_ci(np.array([0.1]), n_boot=0, seed=1)
    rates = validate_mod.genre_high_rates(frame)
    residuals = validate_mod.theme_residuals(frame, rates)
    with pytest.raises(pre.PreconditionError, match="fallback_magnitude"):
        validate_mod.flag_oddballs(frame, rates, residuals, fallback_magnitude=-0.1)
    with pytest.raises(pre.PreconditionError, match="links_path"):
        validate_mod.validate_movielens(frame, tmp_path / "nope.csv", tmp_path / "nope2.csv")


def test_report_contracts(tmp_path: Path) -> None:
    """Report helpers refuse missing artifacts, unknown kinds and tiny clips."""
    with pytest.raises(pre.PreconditionError, match="load_stats"):
        report_mod.load_artifacts(tmp_path)
    with pytest.raises(pre.PreconditionError, match="'kind' must be one of"):
        report_mod._direction_list(pd.DataFrame({"flag": []}), "sideways")
    with pytest.raises(pre.PreconditionError, match="limit"):
        report_mod._clip("abc", limit=1)
    assert report_mod._clip("abcdef", limit=4) == "abc…"
    with pytest.raises(pre.PreconditionError, match="scores"):
        report_mod.make_silhouette_chart({}, tmp_path)
