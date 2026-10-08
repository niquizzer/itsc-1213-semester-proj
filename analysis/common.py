"""Shared helpers for the keyword-theme-analysis pipeline scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import sparse

ANALYSIS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = ANALYSIS_DIR.parent
OUTPUT_DIR: Path = ANALYSIS_DIR / "output"

_loaded: dict[str, types.ModuleType] = {}


def load_script(stem: str) -> types.ModuleType:
    """Import a pipeline script (e.g. '01-load') by file path, cached."""
    if stem in _loaded:
        return _loaded[stem]
    path = ANALYSIS_DIR / f"{stem}.py"
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
    """Create (if needed) and return the pipeline output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: object, path: Path) -> None:
    """Write JSON with sorted keys and UTF-8 encoding."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> object:
    """Read a JSON file written by save_json."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_sparse(matrix: sparse.csr_matrix, ids: np.ndarray, output_dir: Path) -> None:
    """Persist a TF-IDF matrix and its row movie ids as npz artifacts."""
    out = ensure_output_dir(output_dir)
    sparse.save_npz(out / "tfidf.npz", matrix.tocsr())
    np.save(out / "tfidf_ids.npy", ids)


def load_sparse(output_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load a TF-IDF matrix and row ids saved by save_sparse."""
    out = ensure_output_dir(output_dir)
    matrix = sparse.load_npz(out / "tfidf.npz")
    ids = np.load(out / "tfidf_ids.npy", allow_pickle=False)
    return matrix, ids