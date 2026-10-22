"""S9 tests study (Phase 6.5): the quick grid runs, is JSON-able, and reports."""

import json

import helpers as h
import pytest

from sims import runner


def test_s9_quick_cells_and_report(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from sims.studies import s9_tests as s9

    rows = []
    for params in runner.study_grid(s9, quick=True):
        out = s9.run_cell(params, runner.task_seed(s9.NAME, params, 0))
        json.dumps(out)
        rows.append({**params, **out})
    for r in rows:
        for tag in ("reml", "ml"):
            # the exact refit reaches the grid search's and the oracle's maximum
            assert r["short_new_" + tag] <= h.REFIT_LL_ATOL
            assert r["oracle_short_" + tag] <= h.ORACLE_LL_ATOL
            assert r["dh_" + tag] <= h.REFIT_H_ATOL
        for t in s9.TESTS:
            assert 0.3 < r["lambda_" + t] < 3.0
    s9.report(pd.DataFrame(rows), str(tmp_path))
    text = (tmp_path / "summary.md").read_text()
    for part in (
        "## Null calibration",
        "## The LRT",
        "## The tail",
        "## Power",
        "## The exact refit",
    ):
        assert part in text
    assert (tmp_path / "summary.csv").exists()
