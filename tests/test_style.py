"""Design D10/D12: every pipeline function has a docstring, full type hints,
and a documented ``Preconditions:`` section."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from analysis import common

MODULES: list[str] = [
    "01-load", "02-vocab", "03-cluster", "04-ratings", "05-validate", "06-report",
    "common", "preconditions", "sorting",
]


def _function_defs(source: str) -> list[ast.FunctionDef]:
    """Return every FunctionDef node, including nested ones."""
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


@pytest.mark.parametrize("stem", MODULES)
def test_every_function_has_docstring_and_type_hints(stem: str) -> None:
    """Each public function carries a docstring, arg annotations, and return type."""
    path = common.ANALYSIS_DIR / f"{stem}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert functions, f"{stem}.py should define at least one function"
    for node in functions:
        docstring = ast.get_docstring(node)
        assert docstring and docstring.strip(), f"{stem}.py:{node.lineno} {node.name} lacks a docstring"
        for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            assert argument.annotation is not None, (
                f"{stem}.py:{node.lineno} {node.name} argument '{argument.arg}' lacks a type hint"
            )
        assert node.args.vararg is None or node.args.vararg.annotation is not None, (
            f"{stem}.py:{node.lineno} {node.name} *args lacks a type hint"
        )
        assert node.args.kwarg is None or node.args.kwarg.annotation is not None, (
            f"{stem}.py:{node.lineno} {node.name} **kwargs lacks a type hint"
        )
        assert node.returns is not None, (
            f"{stem}.py:{node.lineno} {node.name} lacks a return type hint"
        )


@pytest.mark.parametrize("stem", MODULES)
def test_every_function_documents_its_preconditions(stem: str) -> None:
    """Every function docstring has a ``Preconditions:`` section (D12).

    A function with no preconditions still says so explicitly
    (``Preconditions: none ...``) so a reader never has to guess whether the
    contract was considered or simply forgotten.
    """
    path = common.ANALYSIS_DIR / f"{stem}.py"
    source = path.read_text(encoding="utf-8")
    for node in _function_defs(source):
        docstring = ast.get_docstring(node) or ""
        assert "Preconditions:" in docstring, (
            f"{stem}.py:{node.lineno} {node.name} has no 'Preconditions:' section"
        )


def test_common_helpers_are_documented() -> None:
    """common.py also follows the docstring + type-hint convention."""
    source = (common.ANALYSIS_DIR / "common.py").read_text(encoding="utf-8")
    for node in _function_defs(source):
        docstring = ast.get_docstring(node)
        assert docstring and docstring.strip(), f"common.py:{node.lineno} {node.name} lacks a docstring"
