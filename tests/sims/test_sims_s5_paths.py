"""S5 edge cases: absolute paths in error and warning text are shortened, nothing else is."""

import pytest

from sims.studies.s5_edge_cases import _relpaths


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # POSIX checkout path
        (
            "/opt/clone/pylmm/lmm.py:220: LinAlgWarning: ill-conditioned",
            "pylmm/lmm.py:220: LinAlgWarning: ill-conditioned",
        ),
        # a traceback line, path in quotes, subpackage
        ('  File "/opt/clone/pylmm/cli/gwas.py", line 3', '  File "pylmm/cli/gwas.py", line 3'),
        # Windows
        ("C:\\work\\clone\\sims\\runner.py:12: warn", "sims\\runner.py:12: warn"),
        # a checkout that itself sits under directories named pylmm/
        ("/opt/pylmm/pylmm1/wt/pylmm/lmm.py:250: x", "pylmm/lmm.py:250: x"),
        ("/opt/pylmm/pylmm/lmm.py:1", "pylmm/lmm.py:1"),
        # several paths in one string
        (
            "/a/b/pylmm/lmm.py:1 and /c/d/tests/helpers.py:2",
            "pylmm/lmm.py:1 and tests/helpers.py:2",
        ),
        # installed packages: the part after site-packages
        (
            "/usr/lib/python3.13/site-packages/scipy/optimize/_optimize.py:9: RuntimeWarning",
            "scipy/optimize/_optimize.py:9: RuntimeWarning",
        ),
        ("/venv/lib/python3.13/site-packages/pylmm/lmm.py:220", "pylmm/lmm.py:220"),
    ],
)
def test_absolute_paths_are_shortened(text, expected):
    assert _relpaths(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "see docs/x/pylmm/lmm.py",  # relative path
        "pylmm/lmm.py:220: LinAlgWarning",  # already relative
        "https://github.com/nickFurlotte/pylmm/blob/main/pylmm/lmm.py",  # URL
        "ratio 1/2/pylmm",  # not a path
        "/tmp/foo.txt not found",  # absolute, but not a repo .py path
        "numpy.linalg.LinAlgError: singular matrix",
    ],
)
def test_other_text_is_unchanged(text):
    assert _relpaths(text) == text
