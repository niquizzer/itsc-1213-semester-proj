"""Smoke tests: tooling works and pipeline modules are importable."""

from __future__ import annotations

from analysis import common


def test_smoke_imports() -> None:
    """The shared helper can load every numbered pipeline script."""
    for stem in ("01-load", "02-vocab", "03-cluster", "04-ratings",
                 "05-validate", "06-report"):
        module = common.load_script(stem)
        assert callable(module.main), f"{stem}.main should be callable"


def test_output_dir_is_created(tmp_path) -> None:
    """ensure_output_dir creates and returns the resolved output directory."""
    out = common.ensure_output_dir(tmp_path)
    assert out.exists() and out.is_dir()