"""Precondition helpers: the pipeline's lightweight design-by-contract layer.

A *precondition* is something that must already be true at the moment a
function is called. If it is not true, the function makes no promise about
what happens next. Every pipeline function in this project documents its
preconditions in a ``Preconditions:`` section of its docstring and enforces
them at the very top of its body with the helpers in this module, so a bad
call fails *immediately* with a message that names the offending argument,
instead of producing a confusing traceback (or, worse, a silently wrong
number) somewhere deep inside pandas or scikit-learn.

Conventions
-----------
* Each helper raises :class:`PreconditionError` (a ``ValueError``) when the
  condition fails and simply returns ``None`` otherwise, so calls read like
  plain statements at the top of a function::

      require_columns(themed, ["theme", "high_rating"], "themed")
      require_fraction(confidence, "confidence", exclusive=True)

* Type violations raise :class:`TypePreconditionError`, which inherits from
  both ``PreconditionError`` and ``TypeError``. That means ``except
  TypeError``, ``except ValueError`` and ``except PreconditionError`` all
  catch it, so callers can be as broad or as narrow as they like.
* Messages always include the *name* of the argument and a short description
  of the value that was actually received, so the traceback alone is enough
  to diagnose the call site.
* Helpers never mutate or coerce their inputs. A caller that wants
  ``int(value)`` must do that itself *after* the check passes.
* ``bool`` is deliberately **not** accepted where an integer or real number is
  required, even though ``bool`` subclasses ``int`` in Python. Passing
  ``True`` as a vote floor is almost always a bug.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Mapping, Sized
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

#: Longest ``repr`` fragment we embed in an error message. Long DataFrames or
#: lists would otherwise turn a one-line error into a wall of text.
_REPR_LIMIT: int = 60


class PreconditionError(ValueError):
    """Raised when a function is called with arguments that violate its contract.

    Subclassing ``ValueError`` keeps existing ``except ValueError`` handlers
    working, while giving callers a precise type to catch when they want to
    distinguish "you called me wrong" from "the data is bad".
    """


class TypePreconditionError(PreconditionError, TypeError):
    """Raised when an argument has the wrong *type* (a ``TypeError`` as well).

    Python's convention is that wrong types raise ``TypeError`` and wrong
    values raise ``ValueError``; inheriting from both ``PreconditionError`` and
    ``TypeError`` honours that convention without forcing callers to catch
    two different exception classes.
    """


def _describe(value: object) -> str:
    """Return a short ``type name + repr`` description of ``value`` for messages.

    Preconditions:
        none - any value, including ``None``, can be described.
    """
    text = repr(value)
    if len(text) > _REPR_LIMIT:
        text = text[: _REPR_LIMIT - 3] + "..."
    return f"{type(value).__name__} {text}"


def _type_names(expected: type | tuple[type, ...]) -> str:
    """Join one or more expected type names for an error message.

    Preconditions:
        ``expected`` is a ``type`` or a non-empty tuple of ``type`` objects.
    """
    if isinstance(expected, tuple):
        return " | ".join(t.__name__ for t in expected)
    return expected.__name__


def require(condition: object, message: str) -> None:
    """Raise :class:`PreconditionError` with ``message`` unless ``condition`` is truthy.

    This is the general-purpose escape hatch for conditions that none of the
    more specific helpers below express, e.g. "every genre in the frame must
    have a known rate".

    Preconditions:
        ``message`` is a non-empty string so the error is never blank.
    """
    if not isinstance(message, str) or not message.strip():
        raise TypePreconditionError("require(): 'message' must be a non-empty str")
    if not condition:
        raise PreconditionError(message)


def require_type(value: object, expected: type | tuple[type, ...], name: str) -> None:
    """Require ``isinstance(value, expected)``.

    Preconditions:
        ``expected`` is a type or a tuple of types; ``name`` is the argument's
        name as the caller spells it (used only in the error message).
    """
    if not isinstance(value, expected):
        raise TypePreconditionError(
            f"'{name}' must be {_type_names(expected)}, got {_describe(value)}"
        )


def require_callable(value: object, name: str) -> None:
    """Require that ``value`` can be called (a function, lambda, method, ...).

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    if not callable(value):
        raise TypePreconditionError(f"'{name}' must be callable, got {_describe(value)}")


def require_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    """Require an integer (``int`` or NumPy integer, but **not** ``bool``) in an optional range.

    Both bounds are inclusive. Leave a bound ``None`` to skip that side.

    Preconditions:
        if both bounds are given then ``minimum <= maximum``.
    """
    if minimum is not None and maximum is not None and minimum > maximum:
        raise PreconditionError(
            f"require_int('{name}'): minimum {minimum} exceeds maximum {maximum}"
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypePreconditionError(f"'{name}' must be an int, got {_describe(value)}")
    if minimum is not None and value < minimum:
        raise PreconditionError(f"'{name}' must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise PreconditionError(f"'{name}' must be <= {maximum}, got {value}")


def require_real(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
    allow_nan: bool = False,
) -> None:
    """Require a real number (``int``/``float``/NumPy scalar, not ``bool``) in a range.

    ``exclusive_minimum`` / ``exclusive_maximum`` turn the matching bound into a
    strict inequality. ``NaN`` is rejected unless ``allow_nan`` is set, because
    a ``NaN`` silently passes every ``<``/``>`` comparison and would otherwise
    sneak through any range check.

    Preconditions:
        if both bounds are given then ``minimum <= maximum``.
    """
    if minimum is not None and maximum is not None and minimum > maximum:
        raise PreconditionError(
            f"require_real('{name}'): minimum {minimum} exceeds maximum {maximum}"
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypePreconditionError(f"'{name}' must be a real number, got {_describe(value)}")
    number = float(value)
    if math.isnan(number):
        if allow_nan:
            return
        raise PreconditionError(f"'{name}' must not be NaN")
    if minimum is not None:
        if exclusive_minimum and not number > minimum:
            raise PreconditionError(f"'{name}' must be > {minimum}, got {value}")
        if not exclusive_minimum and not number >= minimum:
            raise PreconditionError(f"'{name}' must be >= {minimum}, got {value}")
    if maximum is not None:
        if exclusive_maximum and not number < maximum:
            raise PreconditionError(f"'{name}' must be < {maximum}, got {value}")
        if not exclusive_maximum and not number <= maximum:
            raise PreconditionError(f"'{name}' must be <= {maximum}, got {value}")


def require_fraction(value: object, name: str, *, exclusive: bool = False) -> None:
    """Require a real number in ``[0, 1]`` (or the open interval ``(0, 1)``).

    Used for confidence levels, document-frequency ceilings, and rates.

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    require_real(
        value, name,
        minimum=0.0, maximum=1.0,
        exclusive_minimum=exclusive, exclusive_maximum=exclusive,
    )


def require_ordered(low: object, high: object, low_name: str, high_name: str) -> None:
    """Require ``low <= high`` for two already-validated numbers.

    Preconditions:
        ``low`` and ``high`` are comparable with ``<=`` (check them with
        :func:`require_int` / :func:`require_real` first).
    """
    if not low <= high:  # type: ignore[operator]
        raise PreconditionError(
            f"'{low_name}' ({low}) must be <= '{high_name}' ({high})"
        )


def require_non_empty(value: Sized, name: str) -> None:
    """Require ``len(value) > 0`` for a list, dict, DataFrame, array, string, ...

    Preconditions:
        ``value`` supports ``len()``.
    """
    if not isinstance(value, Sized):
        raise TypePreconditionError(f"'{name}' must have a length, got {_describe(value)}")
    if len(value) == 0:
        raise PreconditionError(f"'{name}' must not be empty")


def require_same_length(
    first: Sized, second: Sized, first_name: str, second_name: str
) -> None:
    """Require two sized objects to have the same length (parallel lists, ids vs rows, ...).

    Preconditions:
        both values support ``len()``.
    """
    if not isinstance(first, Sized) or not isinstance(second, Sized):
        raise TypePreconditionError(
            f"'{first_name}' and '{second_name}' must both have a length"
        )
    if len(first) != len(second):
        raise PreconditionError(
            f"'{first_name}' (len {len(first)}) and '{second_name}' "
            f"(len {len(second)}) must have the same length"
        )


def require_all(
    values: Iterable[Any],
    predicate: Callable[[Any], bool],
    name: str,
    description: str,
) -> None:
    """Require every element of ``values`` to satisfy ``predicate``.

    ``description`` finishes the sentence "every element of '<name>' must be
    ...", e.g. ``"a non-empty str"``. The first failing element is reported
    with its position.

    Preconditions:
        ``predicate`` is callable; ``values`` is iterable (it is consumed, so
        pass a list rather than a one-shot generator you still need).
    """
    require_callable(predicate, "predicate")
    for position, item in enumerate(values):
        if not predicate(item):
            raise PreconditionError(
                f"every element of '{name}' must be {description}; "
                f"element {position} is {_describe(item)}"
            )


def require_columns(frame: object, columns: Iterable[str], name: str) -> None:
    """Require ``frame`` to be a ``DataFrame`` containing every column in ``columns``.

    An empty frame with the right columns passes; a frame missing even one
    column fails and the message lists *all* missing columns at once.

    Preconditions:
        ``columns`` is an iterable of column names (strings).
    """
    require_type(frame, pd.DataFrame, name)
    wanted = list(columns)
    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        raise PreconditionError(
            f"'{name}' is missing required column(s) {missing}; "
            f"has {list(frame.columns)}"
        )


def require_keys(mapping: object, keys: Iterable[Any], name: str) -> None:
    """Require ``mapping`` to be a ``Mapping`` (dict-like) containing every key in ``keys``.

    Preconditions:
        ``keys`` is iterable.
    """
    require_type(mapping, Mapping, name)
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise PreconditionError(f"'{name}' is missing required key(s) {missing}")


def require_existing_file(path: object, name: str) -> None:
    """Require ``path`` to be a ``Path`` (or ``str``) that names an existing *file*.

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    require_type(path, (Path, str), name)
    candidate = Path(path)
    if not candidate.is_file():
        raise PreconditionError(f"'{name}' must be an existing file, got {candidate}")


def require_existing_dir(path: object, name: str) -> None:
    """Require ``path`` to be a ``Path`` (or ``str``) that names an existing *directory*.

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    require_type(path, (Path, str), name)
    candidate = Path(path)
    if not candidate.is_dir():
        raise PreconditionError(f"'{name}' must be an existing directory, got {candidate}")


def require_sparse_matrix(matrix: object, name: str) -> None:
    """Require a two-dimensional SciPy sparse matrix (any format: CSR, CSC, COO, ...).

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    if not sparse.issparse(matrix):
        raise TypePreconditionError(
            f"'{name}' must be a scipy sparse matrix, got {_describe(matrix)}"
        )
    shape = getattr(matrix, "shape", ())
    if len(shape) != 2:
        raise PreconditionError(f"'{name}' must be 2-dimensional, got shape {shape}")


def require_array_1d(array: object, name: str) -> None:
    """Require a one-dimensional array-like (NumPy array, list, tuple, Series).

    Preconditions:
        none beyond ``name`` being the argument's name.
    """
    shape = getattr(array, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise PreconditionError(f"'{name}' must be 1-dimensional, got shape {shape}")
        return
    if not isinstance(array, (list, tuple)):
        raise TypePreconditionError(
            f"'{name}' must be a 1-d array, list or tuple, got {_describe(array)}"
        )
