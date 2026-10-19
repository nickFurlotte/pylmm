"""S1 null-calibration study: one quick cell runs, is JSON-able, and reports."""

import json

import pytest

from sims import runner


def test_s1_quick_cell_and_report(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    import pandas as pd

    from sims.studies import s1_null_calibration as s1

    params = runner.study_grid(s1, quick=True)[0]  # unrelated, tiny
    rows = []
    for rep in range(2):
        out = s1.run_cell(params, runner.task_seed(s1.NAME, params, rep))
        json.dumps(out)
        rows.append({**params, **out})
    out = rows[0]
    assert out["n_reml"] > 0 and 0 <= out["h_hat"] <= 1
    # OLS and LMM agree on a null cohort without structure
    assert 0.5 < out["lam_ols"] < 1.8 and 0.5 < out["lam_reml"] < 1.8
    s1.report(pd.DataFrame(rows), str(tmp_path))
    for f in ("summary.csv", "summary.md", "qq_panel.png", "lambda_vs_h2.png"):
        assert (tmp_path / f).exists()
