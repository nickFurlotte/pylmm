"""``legacy_fit=True`` / ``pylmm gwas --legacy-fit`` restores pylmm 1.0.0 (Phase 6.3).

The golden tests (test_null_fit.py, test_association.py, test_cli_gwas.py,
test_misc.py) run with the switch against the 2013 outputs, within the
tolerances of tests/helpers.py.  Here the switch is checked against pylmm
1.0.0 itself, byte for byte: the ``v1.0.0`` tag's ``pylmm/`` is extracted with
``git archive`` and its ``pylmm gwas`` is run next to this checkout's
``pylmm gwas --legacy-fit --no-batch`` on the same inputs, on this machine.
The test is skipped where the tag is not available (a shallow clone without
tags).
"""

import os
import subprocess
import sys
import tarfile

import helpers as h
import numpy as np
import pytest

V100 = "v1.0.0"

_RUN = (
    "import sys; d = sys.argv[1]; sys.path.insert(0, d); import pylmm; "
    "assert pylmm.__file__.startswith(d), pylmm.__file__; "
    "from pylmm.cli import main; sys.exit(main(sys.argv[2:]))"
)


@pytest.fixture(scope="module")
def v100(tmp_path_factory):
    """The pylmm/ package of the v1.0.0 tag, extracted into a temporary directory."""
    d = tmp_path_factory.mktemp("v100")
    tar = d / "pylmm.tar"
    try:
        with open(tar, "wb") as f:
            subprocess.run(
                ["git", "archive", "--format=tar", V100, "pylmm"],
                cwd=h.REPO,
                stdout=f,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=60,
            )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("the %s tag is not available (shallow clone?)" % V100)
    with tarfile.open(tar) as t:
        t.extractall(d / "src", filter="data")
    return str(d / "src")


def run(tree, args, cwd):
    """Run ``pylmm`` from ``tree`` (a source directory) with ``args``; the output."""
    p = subprocess.run(
        [sys.executable, "-c", _RUN, tree, *[str(a) for a in args]],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=h.CLI_TIMEOUT,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout + p.stderr


def _bed(tmp_path, nsnps):
    base = h.write_bed_subset(tmp_path, nsnps)
    return ["--bfile", base, "--kinship", h.README_KIN, "--pheno", h.FAKE_PHENOS]


def _emma(tmp_path):
    emma = h.write_emma_subset(tmp_path, 200)
    pheno = h.write_emma_missing_pheno(tmp_path)
    return [
        "--emma-geno",
        emma,
        "--emma-pheno",
        pheno,
        "--kinship",
        h.write_headerless_kinship(tmp_path),
    ]


CASES = {
    # README data: the REML maximum is inside [0, 0.01], where 1.0.0 returned h = 0
    "bed1000": (lambda d: _bed(d, h.BED_GWAS_NSNPS), []),
    "bed50-ml": (lambda d: _bed(d, h.BED_GWAS_SMALL_NSNPS), ["--ml"]),
    "bed50-refit": (lambda d: _bed(d, h.BED_GWAS_SMALL_NSNPS), ["--refit"]),
    # hmdp: missing phenotypes and missing calls (per-SNP fits), SNP 110 bimodal
    "emma200": (_emma, []),
    "emma200-remove-missing": (_emma, ["--remove-missing-genotypes"]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_legacy_fit_no_batch_is_v100_byte_for_byte(v100, tmp_path, case):
    make_inputs, flags = CASES[case]
    inputs = make_inputs(tmp_path)
    old, new, batched = (str(tmp_path / n) for n in ("old.tsv", "new.tsv", "batched.tsv"))
    run(v100, ["gwas", "-q", *inputs, *flags, "-o", old], tmp_path)
    run(h.REPO, ["gwas", "-q", "--legacy-fit", "--no-batch", *inputs, *flags, "-o", new], tmp_path)
    with open(old, "rb") as f_old, open(new, "rb") as f_new:
        assert f_old.read() == f_new.read(), "%s: --legacy-fit --no-batch differs from 1.0.0" % case
    # batched (the default path) with the legacy fit: 1.0.0 up to rounding
    run(h.REPO, ["gwas", "-q", "--legacy-fit", *inputs, *flags, "-o", batched], tmp_path)
    ids_o, got_o = h.load_gwas_out(old)
    ids_b, got_b = h.load_gwas_out(batched)
    assert ids_o == ids_b
    for j in range(4):
        h.assert_batch_close(got_b[:, j], got_o[:, j], what="%s column %d" % (case, j))


def test_v100_tree_is_really_v100(v100):
    """The extracted tree is 1.0.0's, not this checkout's (no legacy_fit there)."""
    with open(os.path.join(v100, "pylmm", "lmm.py")) as f:
        src = f.read()
    assert "legacy_fit" not in src and "def fit(" in src


def test_library_legacy_fit_matches_v100_library(v100, hmdp, tmp_path):
    """LMM(..., legacy_fit=True).fit and gwas(..., legacy_fit=True, batch=False)
    equal 1.0.0's LMM.fit and gwas exactly (REML and ML, missing calls).  Both
    run in a fresh interpreter, so that the BLAS threading is the same (this
    test process may have pinned it to one thread, which moves the last bits)."""
    Y, K, snps = hmdp
    code = (
        "import sys, numpy as np; d = sys.argv[1]; sys.path.insert(0, d); import pylmm; "
        "assert pylmm.__file__.startswith(d), pylmm.__file__; from pylmm.lmm import LMM, gwas; "
        "kw = eval(sys.argv[4]); Y, K, X = (np.load(sys.argv[2])[k] for k in 'YKX'); out = []; "
        "[out.extend([float(LMM(Y, K, **kw).fit(reml=r)[i]) for i in (0, 2, 3)]) "
        "for r in (True, False)]; "
        "kw.update(eval(sys.argv[5])); "
        "[out.extend(np.ravel(gwas(Y, X, K, reml=r, **kw)).tolist()) for r in (True, False)]; "
        "np.save(sys.argv[3], np.array(out))"
    )
    inp = str(tmp_path / "in.npz")
    np.savez(inp, Y=Y, K=K, X=snps[:, :120])
    res = {}
    for tag, tree, kw, kw_gwas in (
        ("old", v100, "{}", "{}"),
        ("new", h.REPO, "{'legacy_fit': True}", "{'batch': False}"),
    ):
        res[tag] = str(tmp_path / ("%s.npy" % tag))
        args = [sys.executable, "-c", code, tree, inp, res[tag], kw, kw_gwas]
        subprocess.run(args, check=True, timeout=600)
    np.testing.assert_array_equal(np.load(res["new"]), np.load(res["old"]))
