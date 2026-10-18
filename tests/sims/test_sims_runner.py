"""Runner (seeding, resume, errors, parallel), metrics, and the example study."""

import json
import os
import textwrap

import numpy as np
import pytest

from sims import methods, metrics, runner

FAKE_STUDY = textwrap.dedent('''
    import os
    import numpy as np
    NAME = "fake"
    GRID = {"a": [1, 2], "b": ["x"]}
    QUICK_GRID = {"a": [1], "b": ["x"]}
    REPS = 3
    QUICK_REPS = 2

    def run_cell(params, seed):
        marker = os.environ.get("FAKE_FAIL_MARKER")
        if marker and os.path.exists(marker) and params["a"] == 2:
            raise RuntimeError("boom")
        rng = np.random.default_rng(seed)
        return {"x": float(rng.random()), "a2": params["a"] * 2, "arr": [1.0, 2.0]}
''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    path = tmp_path / "fake_study.py"
    path.write_text(FAKE_STUDY)
    monkeypatch.setenv("SIMS_RESULTS_DIR", str(tmp_path / "results"))
    runner._LOADED.pop(str(path), None)
    return str(path)


def _rows(name="fake", quick=False):
    return runner.read_rows(runner.raw_path(name, quick))


def test_seed_is_deterministic_and_distinct():
    s = runner.task_seed("st", {"a": 1, "b": 2}, 0)
    assert s == runner.task_seed("st", {"b": 2, "a": 1}, 0)     # key order irrelevant
    assert s != runner.task_seed("st", {"a": 1, "b": 2}, 1)
    assert s != runner.task_seed("other", {"a": 1, "b": 2}, 0)
    # SEED_PARAMS: common random numbers across cells differing in other params
    assert runner.task_seed("st", {"a": 1, "m": "x"}, 0, ["a"]) == \
        runner.task_seed("st", {"a": 1, "m": "y"}, 0, ["a"])
    assert 0 <= s < 2 ** 63


def test_expand_grid():
    g = runner.expand_grid({"a": [1, 2], "b": [3], "c": 5})
    assert g == [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 3, "c": 5}]
    assert runner.expand_grid([{"a": 1}]) == [{"a": 1}]


def test_run_resume_and_quick(fake):
    s = runner.run(fake, verbose=False)
    assert (s["ran"], s["skipped"], s["errors"]) == (6, 0, 0)
    rows = _rows()
    assert sorted((r["params"]["a"], r["rep"]) for r in rows) == \
        [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert all(r["metrics"]["arr"] == [1.0, 2.0] for r in rows)
    # resume: nothing to do
    s = runner.run(fake, verbose=False)
    assert (s["ran"], s["skipped"]) == (0, 6)
    # more reps only runs the new ones; seeds of old ones unchanged
    s = runner.run(fake, reps=4, verbose=False)
    assert (s["ran"], s["skipped"]) == (2, 6)
    # quick results live in a separate file
    s = runner.run(fake, quick=True, verbose=False)
    assert s["ran"] == 2 and len(_rows(quick=True)) == 2
    by = {(r["cell_id"], r["rep"]): r["metrics"]["x"] for r in _rows(quick=True)}
    full = {(r["cell_id"], r["rep"]): r["metrics"]["x"] for r in _rows()}
    for k, v in by.items():
        assert full[k] == v           # same cell + rep -> same seed -> same result
    log = runner.read_rows(os.path.join(os.path.dirname(runner.raw_path("fake", False)),
                                        "runs.jsonl"))
    assert len(log) == 4


def test_errors_are_logged_and_retried(fake, tmp_path, monkeypatch):
    marker = tmp_path / "fail"
    marker.write_text("1")
    monkeypatch.setenv("FAKE_FAIL_MARKER", str(marker))
    s = runner.run(fake, verbose=False)
    assert (s["ran"], s["errors"]) == (3, 3)
    errs = runner.read_rows(runner.raw_path("fake", False).replace(".jsonl", ".errors.jsonl"))
    assert len(errs) == 3 and "boom" in errs[0]["error"]
    marker.unlink()
    s = runner.run(fake, verbose=False)
    assert (s["ran"], s["skipped"], s["errors"]) == (3, 3, 0)


def test_truncated_line_is_skipped(fake):
    runner.run(fake, verbose=False)
    with open(runner.raw_path("fake", False), "a") as f:
        f.write('{"cell_id": "trunc')
    assert len(_rows()) == 6


def test_parallel_matches_serial(fake, tmp_path, monkeypatch):
    runner.run(fake, verbose=False)
    serial = {(r["cell_id"], r["rep"]): r["metrics"] for r in _rows()}
    monkeypatch.setenv("SIMS_RESULTS_DIR", str(tmp_path / "results_par"))
    s = runner.run(fake, jobs=2, chunk=1, verbose=False)
    assert s["jobs"] == 2 and s["ran"] == 6
    par = {(r["cell_id"], r["rep"]): r["metrics"] for r in _rows()}
    assert par == serial


def test_job_cap(fake):
    s = runner.run(fake, quick=True, jobs=64, verbose=False)
    assert s["jobs"] == runner.MAX_JOBS


def test_report_load(fake):
    pd = pytest.importorskip("pandas")
    from sims import report
    runner.run(fake, verbose=False)
    df = report.load_results("fake")
    assert len(df) == 6 and {"a", "b", "x", "arr", "rep"} <= set(df.columns)
    s = report.summarize(df, ["a"], ["x", "a2"])
    assert list(s["n_reps"]) == [3, 3] and "x_se" in s.columns
    out = report.report(fake)
    assert os.path.exists(os.path.join(out, "summary.csv"))
    md = open(os.path.join(out, "summary.md")).read()
    assert md.startswith("# ") and "| a |" in md
    assert isinstance(pd.read_csv(os.path.join(out, "summary.csv")), pd.DataFrame)


# ---------------------------------------------------------------------------
# metrics + OLS baseline
# ---------------------------------------------------------------------------

def test_lambda_and_t1e_on_uniform():
    p = np.random.default_rng(0).random(200000)
    assert abs(metrics.lambda_gc(p) - 1) < 0.01
    t1e = metrics.type1_error(p, (0.05, 1e-3))
    assert set(t1e) == {"t1e_0.05", "t1e_1e-03"}
    assert abs(t1e["t1e_0.05"] - 0.05) < 0.003
    assert np.isnan(metrics.lambda_gc([np.nan]))
    chi2 = np.random.default_rng(1).chisquare(1, 100000) * 1.2
    assert abs(metrics.lambda_gc(chi2=chi2) - 1.2) < 0.03


def test_qq_thinning_and_misc():
    p = np.random.default_rng(0).random(100000)
    e, o = metrics.qq_data(p, max_points=500, keep_top=100)
    assert len(e) <= 520 and e[0] == e.max() and np.isclose(o[0], -np.log10(p.min()))
    lo, hi = metrics.rate_ci(5, 100)
    assert lo < 0.05 < hi
    assert metrics.power([1e-9, 0.5], (5e-8,)) == {"power_5e-08": 0.5}
    c = metrics.compare([1, 2, 3.1], [1, 2, 3])
    assert np.isclose(c["max_abs"], 0.1)
    assert metrics.coverage([0.1, 3.0], [1, 1], 0.0) == 0.5
    s = metrics.estimate_summary([0.4, 0.6], 0.5, prefix="h_")
    assert np.isclose(s["h_bias"], 0) and np.isclose(s["h_rmse"], 0.1)


def test_ols_gwas_matches_lstsq_and_handles_missing():
    from scipy import stats
    rng = np.random.default_rng(0)
    n = 80
    G = rng.binomial(2, 0.3, (n, 5)).astype(float)
    G[:, 4] = 1.0                                   # monomorphic
    G[3, 2] = np.nan                                # one missing call
    Z = rng.standard_normal((n, 1))
    X0 = np.c_[np.ones(n), Z]
    y = rng.standard_normal(n)
    y[7] = np.nan
    t, p = methods.ols_gwas(y, G, X0)
    assert np.isnan(t[4]) and np.isnan(p[4])
    for j in range(4):
        ok = ~np.isnan(y) & ~np.isnan(G[:, j])
        X = np.c_[X0[ok], G[ok, j]]
        b, rss, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        df = ok.sum() - 3
        se = np.sqrt(rss[0] / df * np.linalg.inv(X.T @ X)[-1, -1])
        np.testing.assert_allclose(t[j], b[-1] / se, rtol=1e-10)
        np.testing.assert_allclose(p[j], 2 * stats.t.sf(abs(b[-1] / se), df), rtol=1e-8)


# ---------------------------------------------------------------------------
# example study
# ---------------------------------------------------------------------------

def test_example_study_quick_cell():
    from sims.studies import example_null as ex
    params = runner.study_grid(ex, quick=True)[1]
    out = ex.run_cell(params, runner.task_seed("example_null", params, 0))
    json.dumps(out)
    assert 0 <= out["h_hat"] <= 1
    assert 0.5 < out["lambda_lmm"] < 2 and 0.5 < out["lambda_ols"] < 2
    # pylmm agrees with the oracle on this easy problem
    assert abs(out["dh_oracle"]) < 1e-4
    assert abs(out["dll_oracle"]) < 1e-6
    assert out["t_vs_oracle_max_abs"] < 1e-4
    assert len(out["qq_p_lmm"]) == ex.N_QQ_KEEP


def test_plots_write_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from sims import plots
    p = np.random.default_rng(0).random(3000)
    a = plots.qq({"a": p, "b": p ** 1.1}, str(tmp_path / "qq.png"), title="t")
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0.1, 0.9, 0.2, 1.0],
                       "y_se": [0.01] * 4, "g": [1, 1, 2, 2]})
    b = plots.lines(df, "x", "y", str(tmp_path / "l.png"), hue="g", yerr="y_se",
                    identity=True)
    c = plots.scatter([1, 2], [1, 2.1], str(tmp_path / "s.png"))
    for f in (a, b, c):
        assert os.path.getsize(f) > 1000
