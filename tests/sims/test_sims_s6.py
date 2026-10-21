"""S6 kinship-kinds study: the quick cells run, are JSON-able, and report; inbred_panel."""

import json

import numpy as np
import pytest

from sims import genotypes, runner


def test_inbred_panel_generator():
    g = genotypes.inbred_panel(30, 400, n_chrom=4, missing=0.05, seed=3)
    called = g.G[~np.isnan(g.G)]
    assert set(np.unique(called)) <= {0.0, 1.0}
    assert 0.02 < np.isnan(g.G).mean() < 0.08
    assert list(np.unique(g.meta["chrom"])) == [1, 2, 3, 4]
    np.testing.assert_array_equal(
        genotypes.inbred_panel(30, 400, n_chrom=4, seed=3).G[:2],
        genotypes.inbred_panel(30, 400, n_chrom=4, seed=3).G[:2],
    )


def test_s6_quick_cells_and_report(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from sims.studies import s6_kinship_kinds as s6

    rows = []
    for params in runner.study_grid(s6, quick=True):
        out = s6.run_cell(params, runner.task_seed(s6.NAME, params, 0))
        json.dumps(out)
        rows.append({**params, **out})
    for out in rows:
        for k in ("std", "cen", "ibs", "std_loco", "ibs_loco"):
            assert 0 <= out["h_" + k] < 1 and 0.5 < out["lam_" + k] < 1.6
        # the IBS kinship's diagonal is 1; its centred scale is well below 1
        assert out["meandiag_ibs"] == pytest.approx(1.0) and out["cmeandiag_ibs"] < 0.8
    h2 = rows[-1]  # h2 = 0.5: the PVE of every kind is close, their h are not
    assert abs(h2["pve_std"] - h2["pve_ibs"]) < 0.1 and h2["h_ibs"] - h2["h_std"] > 0.1
    s6.report(pd.DataFrame(rows), str(tmp_path))
    assert (tmp_path / "summary.md").exists()
