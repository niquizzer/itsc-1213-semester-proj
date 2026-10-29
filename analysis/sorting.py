"""Custom sort keys for the pipeline: one named key function per ordering rule.

How sorting with a custom key works
-----------------------------------
``sorted(items, key=f)`` and ``list.sort(key=f)`` do not compare the items
themselves. They call ``f(item)`` once per item and compare *those* results.
A key function therefore turns "sort these movies by X" into a small, named,
testable rule. Three idioms cover almost every ordering this project needs:

1. **Tuples break ties.** Python compares tuples element by element, so the
   key ``(-mean_weight, keyword)`` sorts by weight first and, only when two
   weights are equal, alphabetically by keyword. Ties are then deterministic
   instead of depending on input order.
2. **Negate to sort descending.** ``sorted`` is ascending; negating a numeric
   component (``-high_pct``) flips just that component while leaving the
   tie-breakers ascending. ``reverse=True`` would flip *every* component.
3. **Rank tables order categories.** Strings sort alphabetically, which puts
   ``"HIGH"`` before ``"LOW"`` before ``"baseline"`` - meaningless. A dict
   like :data:`CLASSIFICATION_RANK` maps each label to the position we
   actually want, and the key returns that integer.

Every key here follows the same shape: it accepts one *record* - either a
``dict`` or a namedtuple-like row from ``DataFrame.itertuples()`` - and
returns a plain tuple. :func:`sort_frame` applies such a key to each row of a
``DataFrame``, which is how the stage scripts replace ad-hoc ``sort_values``
calls with the named rules below. All sorts are **stable**: records whose keys
compare equal keep their original relative order.

``NaN`` handling: a ``NaN`` compares false against everything, which makes a
sort containing one ill-defined. Every numeric key routes values through
:func:`ascending` / :func:`descending`, which map ``NaN`` to ``+inf`` so that
missing values always land at the *end* of the sorted output, mirroring
pandas' ``na_position="last"`` default.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from analysis.preconditions import (
    require,
    require_all,
    require_callable,
    require_non_empty,
    require_same_length,
    require_type,
)

#: Display/report order of the HIGH-LOW classification produced by stage 04.
#: Lower rank sorts first. Unknown labels get :data:`UNKNOWN_RANK` (last).
CLASSIFICATION_RANK: dict[str, int] = {
    "HIGH": 0,
    "baseline": 1,
    "LOW": 2,
    "Excluded (small)": 3,
}

#: Display/report order of the oddball flag produced by stage 05.
ODDBALL_FLAG_RANK: dict[str, int] = {
    "oddball up": 0,
    "oddball down": 1,
    "not an oddball": 2,
    "Excluded (small)": 3,
}

#: Rank given to any label missing from a rank table, so it sorts after all
#: known labels instead of raising ``KeyError`` in the middle of a sort.
UNKNOWN_RANK: int = 99


def field(record: Any, name: str) -> Any:
    """Read attribute-or-key ``name`` from a dict *or* a namedtuple-like row.

    Stage scripts build rows as ``dict`` objects before turning them into a
    ``DataFrame``, and read them back as namedtuples via ``itertuples()``.
    One accessor lets every key function below work on both shapes.

    Preconditions:
        ``name`` is a non-empty ``str``; ``record`` is a ``Mapping`` holding
        ``name`` as a key, or any object exposing ``name`` as an attribute.
    """
    require_type(name, str, "name")
    require_non_empty(name, "name")
    if isinstance(record, Mapping):
        require(name in record, f"record is missing key '{name}'")
        return record[name]
    require(hasattr(record, name), f"record has no field '{name}'")
    return getattr(record, name)


def ascending(value: Any) -> float:
    """Return ``value`` as a float for an ascending sort, with ``NaN`` last.

    Preconditions:
        ``value`` is convertible with ``float()`` (int, float, NumPy scalar,
        numeric string).
    """
    number = float(value)
    return math.inf if math.isnan(number) else number


def descending(value: Any) -> float:
    """Return ``-value`` as a float for a descending sort, with ``NaN`` still last.

    Preconditions:
        ``value`` is convertible with ``float()``.
    """
    number = float(value)
    return math.inf if math.isnan(number) else -number


def rank_of(label: object, table: Mapping[str, int]) -> int:
    """Look up ``label`` in a rank table; unknown labels rank :data:`UNKNOWN_RANK`.

    Preconditions:
        ``table`` is a ``Mapping`` from label to integer rank.
    """
    require_type(table, Mapping, "table")
    return table.get(str(label), UNKNOWN_RANK)


# --------------------------------------------------------------------------
# Record keys: each takes one row (dict or itertuples row) and returns a tuple.
# --------------------------------------------------------------------------


def theme_id_key(record: Any) -> tuple[int]:
    """Order themes by their integer id, ascending (the canonical table order).

    Preconditions:
        ``record`` has a ``theme`` field convertible with ``int()``.
    """
    return (int(field(record, "theme")),)


def high_pct_key(record: Any) -> tuple[float, int]:
    """Order themes by high-rate ascending (lowest first), ties by theme id.

    Preconditions:
        ``record`` has numeric ``high_pct`` and integer ``theme`` fields.
    """
    return (ascending(field(record, "high_pct")), int(field(record, "theme")))


def high_pct_desc_key(record: Any) -> tuple[float, int]:
    """Order themes by high-rate descending (highest first), ties by theme id.

    Preconditions:
        ``record`` has numeric ``high_pct`` and integer ``theme`` fields.
    """
    return (descending(field(record, "high_pct")), int(field(record, "theme")))


def residual_key(record: Any) -> tuple[float, int]:
    """Order themes by genre-expected residual ascending (most negative first).

    Preconditions:
        ``record`` has numeric ``residual`` and integer ``theme`` fields.
    """
    return (ascending(field(record, "residual")), int(field(record, "theme")))


def residual_desc_key(record: Any) -> tuple[float, int]:
    """Order themes by genre-expected residual descending (most positive first).

    Preconditions:
        ``record`` has numeric ``residual`` and integer ``theme`` fields.
    """
    return (descending(field(record, "residual")), int(field(record, "theme")))


def residual_magnitude_key(record: Any) -> tuple[float, int]:
    """Order themes by ``|residual|`` descending: the biggest surprise first.

    Preconditions:
        ``record`` has numeric ``residual`` and integer ``theme`` fields.
    """
    residual = float(field(record, "residual"))
    magnitude = math.nan if math.isnan(residual) else abs(residual)
    return (descending(magnitude), int(field(record, "theme")))


def count_desc_key(record: Any) -> tuple[float, int]:
    """Order themes by member count descending (biggest theme first).

    Preconditions:
        ``record`` has numeric ``count`` and integer ``theme`` fields.
    """
    return (descending(field(record, "count")), int(field(record, "theme")))


def classification_rank_key(record: Any) -> tuple[int, float, int]:
    """Order themes HIGH, then baseline, then LOW, then excluded; within a group by high-rate descending.

    This is the "leaderboard" order used when printing stage 04 results.

    Preconditions:
        ``record`` has ``classification`` (str), numeric ``high_pct`` and
        integer ``theme`` fields.
    """
    return (
        rank_of(field(record, "classification"), CLASSIFICATION_RANK),
        descending(field(record, "high_pct")),
        int(field(record, "theme")),
    )


def oddball_flag_rank_key(record: Any) -> tuple[int, float, int]:
    """Order themes up, then down, then not-oddball, then excluded; within a group by ``|residual|`` descending.

    Preconditions:
        ``record`` has ``flag`` (str), numeric ``residual`` and integer
        ``theme`` fields.
    """
    rank = rank_of(field(record, "flag"), ODDBALL_FLAG_RANK)
    magnitude, theme = residual_magnitude_key(record)
    return (rank, magnitude, theme)


def genre_count_key(record: Any) -> tuple[float, str]:
    """Order a theme's genres by mention count descending, ties alphabetically.

    Preconditions:
        ``record`` has numeric ``count`` and string ``genre`` fields.
    """
    return (descending(field(record, "count")), str(field(record, "genre")))


# --------------------------------------------------------------------------
# Key *factories*: build a key that closes over data the key needs to consult.
# --------------------------------------------------------------------------


def keyword_weight_key(
    means: Sequence[float], vocab: Sequence[str]
) -> Callable[[int], tuple[float, str]]:
    """Build a key that ranks vocabulary *column indices* by mean TF-IDF weight.

    The returned key maps a column index ``j`` to ``(-means[j], vocab[j])``:
    heaviest keyword first, alphabetical among equal weights. Sorting
    ``range(len(vocab))`` with it yields the index order used to label themes.

    Preconditions:
        ``means`` and ``vocab`` are the same length; ``vocab`` entries are
        strings.
    """
    require_same_length(means, vocab, "means", "vocab")
    require_all(list(vocab), lambda term: isinstance(term, str), "vocab", "a str")

    def key(column: int) -> tuple[float, str]:
        """Key for one vocabulary column index.

        Preconditions:
            ``0 <= column < len(vocab)``.
        """
        require(0 <= column < len(vocab), f"column {column} out of range for vocab")
        return (descending(means[column]), vocab[column])

    return key


def silhouette_key(scores: Mapping[int, float]) -> Callable[[int], tuple[float, int]]:
    """Build a key for choosing the best ``k`` from a ``{k: silhouette}`` table.

    Designed for ``max(scores, key=silhouette_key(scores))``: the key is
    ``(score, -k)``, so the highest silhouette wins and, when two values of
    ``k`` tie exactly, the *smaller* (simpler) ``k`` is preferred.

    Preconditions:
        ``scores`` is a non-empty mapping of int ``k`` to numeric score.
    """
    require_type(scores, Mapping, "scores")
    require_non_empty(scores, "scores")

    def key(k: int) -> tuple[float, int]:
        """Key for one candidate ``k``.

        Preconditions:
            ``k`` is a key of ``scores``.
        """
        require(k in scores, f"k={k} is not a candidate in scores")
        return (ascending(scores[k]), -int(k))

    return key


# --------------------------------------------------------------------------
# Applying a record key to a DataFrame.
# --------------------------------------------------------------------------


def sort_frame(
    frame: pd.DataFrame,
    key: Callable[[Any], Any],
    reverse: bool = False,
) -> pd.DataFrame:
    """Return ``frame`` with rows stably reordered by ``key(row)``.

    Each row is handed to ``key`` as the namedtuple produced by
    ``frame.itertuples(index=False)``, so any record key in this module works
    directly. The original index labels travel with their rows (like
    ``sort_values``); call ``.reset_index(drop=True)`` if you want 0..n-1.
    An empty frame is returned unchanged (as a copy).

    Preconditions:
        ``frame`` is a ``DataFrame``; ``key`` is callable; ``reverse`` is a
        ``bool``. Column names must be valid identifiers for ``itertuples``
        to expose them by name (true for every artifact in this pipeline).
    """
    require_type(frame, pd.DataFrame, "frame")
    require_callable(key, "key")
    require_type(reverse, bool, "reverse")
    if frame.empty:
        return frame.copy()
    rows = list(frame.itertuples(index=False))
    # Sort positions, not rows: the key sees the row, the result is the
    # position, and iloc rebuilds the frame in that order. Python's sort is
    # stable, so equal keys keep their original relative order.
    order = sorted(range(len(rows)), key=lambda position: key(rows[position]), reverse=reverse)
    return frame.iloc[order]


def sort_records(
    records: Sequence[Any],
    key: Callable[[Any], Any],
    reverse: bool = False,
) -> list[Any]:
    """Return a new list of ``records`` stably sorted by ``key``.

    A thin, precondition-checked wrapper around ``sorted`` so list-of-dict
    sorts in the stage scripts read the same way as :func:`sort_frame`.

    Preconditions:
        ``records`` is a sequence (list/tuple); ``key`` is callable;
        ``reverse`` is a ``bool``.
    """
    require_type(records, Sequence, "records")
    require_callable(key, "key")
    require_type(reverse, bool, "reverse")
    return sorted(records, key=key, reverse=reverse)
