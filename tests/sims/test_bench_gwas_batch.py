"""sims/bench/gwas_batch.py (the Phase 6.1 benchmark) runs and its two paths agree."""

import contextlib
import io

import helpers as h
import pytest

from sims.bench import gwas_batch


def test_run_synthetic_agrees():
    r = gwas_batch.run_synthetic(80, 200, 2, reml=True, repeats=1)
    assert r["q"] == 3 and r["per_snp_s"] > 0 and r["batched_s"] > 0
    for key in ("t", "p", "beta", "se"):
        _abs, rel, same_nan = r[key]
        assert same_nan and rel < 1e3 * h.BATCH_RTOL, (key, r[key])


def test_quick_main_without_cli():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gwas_batch.main(["--quick", "--no-cli", "--threads", "1"])
    text = out.getvalue()
    assert "### gwas(): batched vs per SNP" in text and "| 300 | 2000 | 4 | ML |" in text


@pytest.mark.skipif(
    not gwas_batch.os.path.exists(gwas_batch.BED_BASE + ".bed"), reason="no bundled data"
)
def test_run_cli_slice(tmp_path):
    r = gwas_batch.run_cli(30, 1, str(tmp_path))
    assert r is not None and r["m"] == 30
    for _abs, rel, same_nan in r["cols"].values():
        assert same_nan and rel < 1e3 * h.BATCH_RTOL


@pytest.mark.skipif(
    not gwas_batch.os.path.exists(gwas_batch.BED_BASE + ".bed"), reason="no bundled data"
)
def test_run_cli_slice_with_baseline(tmp_path):
    """A baseline checkout (here this one): timed, and its output is byte-identical."""
    r = gwas_batch.run_cli(30, 1, str(tmp_path), baseline=h.REPO)
    assert r is not None and r["baseline"]["s"] > 0 and r["baseline"]["identical"]


@pytest.mark.skipif(
    not gwas_batch.os.path.exists(gwas_batch.BED_BASE + ".bed"), reason="no bundled data"
)
def test_run_cli_baseline_without_pylmm_is_an_error(tmp_path):
    """A baseline directory that is not a pylmm checkout would time this tree again: refused."""
    empty = tmp_path / "not-a-checkout"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="imported from"):
        gwas_batch.run_cli(30, 1, str(tmp_path), baseline=str(empty))
