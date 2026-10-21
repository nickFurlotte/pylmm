"""S8 LOCO study and the LOCO benchmark: the quick runs work and report."""

import contextlib
import io
import json

import pytest

from sims import runner


def test_s8_quick_cells_and_report(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from sims.studies import s8_loco as s8

    rows = []
    for params in runner.study_grid(s8, quick=True):
        out = s8.run_cell(params, runner.task_seed(s8.NAME, params, 0))
        json.dumps(out)
        rows.append({**params, **out})
    for out in rows:
        for name in ("all", "loco"):
            assert 0 <= out["h_" + name] <= 1 and 0.3 < out["lam_" + name] < 2
            assert out["qtl_nlp_" + name] >= 0 and out["causal_chi2_" + name] > 0
    designs = {out["design"] for out in rows}
    assert designs == {"split", "all"}
    for out in rows:  # the iid-null calibration check of the "all" design
        if out["design"] == "all":
            assert 0.3 < out["lam_iid_all"] < 2 and 0.3 < out["lam_iid_loco"] < 2
        else:
            assert "lam_iid_loco" not in out
    s8.report(pd.DataFrame(rows), str(tmp_path))
    assert "LOCO" in (tmp_path / "summary.md").read_text()


def test_bench_loco_quick():
    from sims.bench import loco

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        loco.main(["--quick", "--threads", "1"])
    text = out.getvalue()
    assert "| pylmm gwas --loco |" in text and "over 3 chromosomes" in text
    assert "| 1 | snp101 |" in text  # the planted QTL is the top hit of both
