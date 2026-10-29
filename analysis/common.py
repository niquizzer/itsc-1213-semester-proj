"""Shared helpers for the keyword-theme-analysis pipeline scripts.

Every helper here states its preconditions in a ``Preconditions:`` docstring
section and enforces them with :mod:`analysis.preconditions` before doing any
work, so a wrong path or a mismatched matrix/ids pair fails loudly at the call
site rather than halfway through a file write.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import sparse

from analysis.preconditions import (
    require,
    require_array_1d,
    require_existing_dir,
    require_existing_file,
    require_non_empty,
    require_sparse_matrix,
    require_type,
)

ANALYSIS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = ANALYSIS_DIR.parent
OUTPUT_DIR: Path = ANALYSIS_DIR / "output"

#: File names of the two artifacts written by :func:`save_sparse`.
TFIDF_MATRIX_FILE: str = "tfidf.npz"
TFIDF_IDS_FILE: str = "tfidf_ids.npy"

_loaded: dict[str, types.ModuleType] = {}


def load_script(stem: str) -> types.ModuleType:
    """Import a pipeline script (e.g. ``'01-load'``) by file path, cached.

    The numbered scripts have dashes in their names, so they cannot be
    imported with a normal ``import`` statement; this loads them through
    ``importlib`` under a synthetic module name and caches the result so the
    tests share one module object per stage.

    Preconditions:
        ``stem`` is a non-empty ``str`` and ``analysis/<stem>.py`` exists.
    """
    require_type(stem, str, "stem")
    require_non_empty(stem, "stem")
    if stem in _loaded:
        return _loaded[stem]
    path = ANALYSIS_DIR / f"{stem}.py"
    require(path.is_file(), f"no pipeline script named '{stem}' at {path}")
    module_name = f"pipeline_{stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _loaded[stem] = module
    return module


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create (if needed) and return the pipeline output directory.

    Preconditions:
        ``output_dir`` is a ``Path``; its parent directories may or may not
        exist (they are created).
    """
    require_type(output_dir, Path, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: object, path: Path) -> None:
    """Write JSON with sorted keys and UTF-8 encoding.

    Keys are sorted so re-running a stage produces a byte-identical file
    when the numbers have not changed, which keeps ``git diff`` meaningful.

    Preconditions:
        ``path`` is a ``Path`` whose parent directory exists; ``data`` is
        JSON-serialisable (dicts with string keys, lists, str, int, float,
        bool, None). NumPy scalars are *not* JSON-serialisable - convert them
        with ``int()`` / ``float()`` first.
    """
    require_type(path, Path, "path")
    require_existing_dir(path.parent, "path.parent")
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> object:
    """Read a JSON file written by :func:`save_json`.

    Preconditions:
        ``path`` is a ``Path`` to an existing file containing valid JSON.
    """
    require_type(path, Path, "path")
    require_existing_file(path, "path")
    return json.loads(path.read_text(encoding="utf-8"))


def save_sparse(matrix: sparse.csr_matrix, ids: np.ndarray, output_dir: Path) -> None:
    """Persist a TF-IDF matrix and its row movie ids as npz/npy artifacts.

    Preconditions:
        ``matrix`` is a 2-d SciPy sparse matrix; ``ids`` is a 1-d array with
        exactly one entry per matrix row (``len(ids) == matrix.shape[0]``);
        ``output_dir`` is a ``Path`` (created if missing).
    """
    require_sparse_matrix(matrix, "matrix")
    require_array_1d(ids, "ids")
    require(
        len(ids) == matrix.shape[0],
        f"'ids' has {len(ids)} entries but 'matrix' has {matrix.shape[0]} rows",
    )
    out = ensure_output_dir(output_dir)
    sparse.save_npz(out / TFIDF_MATRIX_FILE, matrix.tocsr())
    np.save(out / TFIDF_IDS_FILE, ids)


def load_sparse(output_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load a TF-IDF matrix and row ids saved by :func:`save_sparse`.

    Preconditions:
        ``output_dir`` is a ``Path`` to a directory holding both
        ``tfidf.npz`` and ``tfidf_ids.npy`` (i.e. stage 02 has been run).
    """
    require_type(output_dir, Path, "output_dir")
    out = ensure_output_dir(output_dir)
    require_existing_file(out / TFIDF_MATRIX_FILE, TFIDF_MATRIX_FILE)
    require_existing_file(out / TFIDF_IDS_FILE, TFIDF_IDS_FILE)
    matrix = sparse.load_npz(out / TFIDF_MATRIX_FILE)
    ids = np.load(out / TFIDF_IDS_FILE, allow_pickle=False)
    return matrix, ids
