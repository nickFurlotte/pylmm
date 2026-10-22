"""sims/bench/refit.py (the Phase 6.5 benchmark) runs and the two refits agree."""

import contextlib
import io

import helpers as h

from sims.bench import refit


def test_run_synthetic_agrees():
    r = refit.run_synthetic(80, 60, 1, reml=True, repeats=1)
    assert r["q"] == 2 and r["per_snp_s"] > 0 and r["exact_s"] > 0
    assert r["dh_max"] <= h.REFIT_H_ATOL
    assert r["short_new"] <= h.REFIT_LL_ATOL and r["short_old"] <= h.REFIT_LL_ATOL
    assert r["dt_rel"] <= h.REFIT_ATOL_SCALE and r["fallbacks"] == 0


def test_quick_main():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        refit.main(["--quick"])
    text = out.getvalue()
    assert "### gwas(refit=True): exact" in text and "| 200 | 300 | 3 | ML |" in text
    assert "### one gwas() scan per test" in text
