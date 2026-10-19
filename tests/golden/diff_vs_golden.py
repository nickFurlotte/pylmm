#!/usr/bin/env python3
"""
Measure how far the current (Python 3) pylmm is from the Phase 1 goldens.

Re-runs every section of generate_golden.py with the current interpreter and
library into a scratch directory, then compares each regenerated file with
the stored golden numerically and prints a Markdown table of max abs / max
rel differences (one row per file, or per column where that is informative).

    uv run python tests/golden/diff_vs_golden.py [--keep DIR]

This never writes to tests/golden/.  It is a reporting tool for
docs/phase3-port.md, not a test; the pass/fail tolerances live in
tests/helpers.py.

Py3 shims applied here (generate_golden.py itself stays the py2 generator):
  * np.load of the py2 .npdump pickles needs encoding='latin1';
  * the CLI runner decodes the -v log to str (py2 Popen output was already
    str), and a failing CLI run aborts with its output, as in the generator.
"""

import argparse
import functools
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

os.environ["PYLMM_GOLDEN_ANY_VERSION"] = "1"

_np_load = np.load


@functools.wraps(_np_load)
def _load_py2_pickles(*a, **kw):
    kw.setdefault("encoding", "latin1")
    return _np_load(*a, **kw)


np.load = _load_py2_pickles

import generate_golden as gen  # noqa: E402


def run_cli(args, outfile, cwd):
    """py3 version of generate_golden.run_cli (same checks, text output)."""
    gen.check_no_comment_header(args)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, *args, outfile]
    p = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        raise RuntimeError("command failed (%d): %s" % (p.returncode, " ".join(cmd)))
    return p.stdout


gen.run_cli = run_cli

_savetxt = gen.savetxt


def savetxt(name, arr, header):
    # association(return_beta=True) returns beta_var as a (1,1) ndarray (sigma
    # is 1x1).  numpy 1.16 coerced the list of mixed rows; numpy >= 1.24
    # refuses ragged input, so flatten each entry to a float first.
    if isinstance(arr, list):
        arr = [[float(np.asarray(v).sum()) for v in row] for row in arr]
    return _savetxt(name, arr, header)


gen.savetxt = savetxt


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_any(fn):
    """Return {column_label: 1-D float array} for a golden-format file."""
    name = os.path.basename(fn)
    if name.endswith(".kv.txt"):
        out = {}
        for line in open(fn):
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split()
                out[k] = np.array([float(v)])
        return out
    if name.endswith(".out"):
        lines = open(fn).read().splitlines()
        cols = lines[0].split("\t")[1:]
        vals = np.array([[float(x) for x in line.split("\t")[1:]] for line in lines[1:]])
        vals = vals.reshape(-1, len(cols))
        return {c: vals[:, j] for j, c in enumerate(cols)}
    a = np.loadtxt(fn)
    header = [ln[2:] for ln in open(fn) if ln.startswith("# columns:")]
    if a.ndim == 2 and header and len(header[0].replace(",", " ").split()) - 1 >= a.shape[1]:
        labels = header[0].replace(",", " ").split()[1:]
        if len(labels) == a.shape[1]:
            return {lab: a[:, j] for j, lab in enumerate(labels)}
    if a.ndim == 2 and name.endswith("diag_Kv.txt"):
        return {"diag": a[:, 0], "K.dot(cos)": a[:, 1]}
    return {"all": a.ravel()}


def diffs(got, exp):
    got = np.asarray(got, float)
    exp = np.asarray(exp, float)
    if got.shape != exp.shape:
        return {"err": "shape %s vs %s" % (got.shape, exp.shape)}
    gn, en = np.isnan(got), np.isnan(exp)
    nanmis = int((gn != en).sum())
    ok = ~gn & ~en
    d = np.abs(got[ok] - exp[ok])
    if d.size == 0:
        return {
            "n": int(exp.size),
            "nan": int(en.sum()),
            "nanmis": nanmis,
            "abs": 0.0,
            "rel": 0.0,
            "exact": bool(nanmis == 0),
        }
    denom = np.abs(exp[ok])
    nz = denom > 0
    rel = (d[nz] / denom[nz]).max() if nz.any() else 0.0
    return {
        "n": int(exp.size),
        "nan": int(en.sum()),
        "nanmis": nanmis,
        "abs": float(d.max()),
        "rel": float(rel),
        "exact": bool(nanmis == 0 and d.max() == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="write regenerated files here and keep them")
    opts = ap.parse_args()
    out = opts.keep or tempfile.mkdtemp(prefix="pylmm_py3_golden_")
    if not os.path.isdir(out):
        os.makedirs(out)
    gen.OUT = os.path.abspath(out)
    os.chdir(REPO)
    t0 = time.time()
    gen.write_env()
    Y, K, snps = gen.load_hmdp()
    gen.golden_kinship(Y, K, snps)
    gen.golden_null_fits(Y, K, snps)
    gen.golden_association(Y, K, snps)
    gen.golden_example_py()
    gen.golden_misc(Y, K, snps)
    work = tempfile.mkdtemp(prefix="pylmm_golden_work_")
    try:
        gen.golden_cli(work, K)
    finally:
        shutil.rmtree(work)
    elapsed = time.time() - t0

    print("# py3 vs py2 goldens")
    print()
    print(
        "python %s, numpy %s, scipy %s; regenerated in %.1fs"
        % (sys.version.split()[0], np.__version__, scipy.__version__, elapsed)
    )
    print()
    print("| file | column | n | nan | nan mismatches | max abs diff | max rel diff |")
    print("|---|---|---|---|---|---|---|")
    for name in sorted(os.listdir(HERE)):
        if not (name.endswith(".txt") or name.endswith(".out")) or name == "env.kv.txt":
            continue
        fn = os.path.join(out, name)
        if not os.path.exists(fn):
            print("| %s | | | | | not regenerated | |" % name)
            continue
        exp, got = read_any(os.path.join(HERE, name)), read_any(fn)
        rows = []
        if name.endswith(".kv.txt"):
            # one row per config prefix
            groups = {}
            for k in exp:
                groups.setdefault(k.split(".")[0] if "." in k else "(all)", []).append(k)
            for g, keys in sorted(groups.items()):
                ge = np.concatenate([exp[k] for k in keys])
                gg = np.concatenate([got.get(k, np.array([np.nan])) for k in keys])
                rows.append((g, diffs(gg, ge)))
        else:
            for c in exp:
                rows.append(
                    (c, diffs(got.get(c), exp[c]) if c in got else {"err": "missing column"})
                )
            if "eigenvalues" in name or name.endswith(".kva.txt"):
                e = exp["all"]
                big = e > 1e-6
                rows.append(("entries > 1e-6", diffs(np.sort(got["all"])[big], e[big])))
        for c, d in rows:
            if "err" in d:
                print("| %s | %s | | | | %s | |" % (name, c, d["err"]))
                continue
            tag = " (bit-identical)" if d["exact"] else ""
            print(
                "| %s | %s | %d | %d | %d | %.3g%s | %.3g |"
                % (name, c, d["n"], d["nan"], d["nanmis"], d["abs"], tag, d["rel"])
            )
    if not opts.keep:
        shutil.rmtree(out)


if __name__ == "__main__":
    main()
