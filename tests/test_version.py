"""pylmm.__version__: from the installed metadata, or from pyproject.toml in a source checkout."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pylmm

REPO = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    for line in (REPO / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("no version line in pyproject.toml")


def test_version_matches_pyproject():
    assert pylmm.__version__ == _pyproject_version()


def test_source_tree_fallback_without_metadata(monkeypatch):
    """Imported from a checkout that is not installed: read pyproject.toml, not 'unknown'."""

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        mod = importlib.reload(pylmm)
        assert mod.__version__ == _pyproject_version()
    finally:
        monkeypatch.undo()
        importlib.reload(pylmm)
    assert pylmm.__version__ == _pyproject_version()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[project]\nname = "pylmm"\nversion = "1.2.3"\n', "1.2.3"),
        ('[project]\nversion = "1.2.3"\nname = "pylmm"\n', "1.2.3"),
        ('[tool.x]\nversion = "9"\n[project]\nname = "pylmm"\nversion = "0.5"\n', "0.5"),
        ('[project]\nname = "other"\nversion = "1.0"\n', "unknown"),
        ('[project]\nname = "pylmm"\ndynamic = ["version"]\n', "unknown"),
    ],
)
def test_source_tree_version_parser(tmp_path, text, expected):
    f = tmp_path / "pyproject.toml"
    f.write_text(text)
    assert pylmm._source_tree_version(f) == expected
    assert pylmm._source_tree_version(tmp_path / "missing.toml") == "unknown"


def test_pythonpath_import_reports_the_version(tmp_path):
    """The replication's case: `PYTHONPATH=<checkout> python -c 'import pylmm'`, not installed.

    A copy of the package and pyproject.toml in a fresh directory, first on sys.path; the
    metadata lookup is made to fail, as it does when pylmm is not installed.
    """
    import shutil

    shutil.copytree(
        REPO / "pylmm", tmp_path / "pylmm", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    code = (
        "import importlib.metadata as md, sys\n"
        "def v(name): raise md.PackageNotFoundError(name)\n"
        "md.version = v\n"
        "sys.path.insert(0, %r)\n"
        "import pylmm; print(pylmm.__file__); print(pylmm.__version__)\n" % str(tmp_path)
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    ).stdout.split()
    assert out[0].startswith(str(tmp_path))
    assert out[1] == _pyproject_version()
