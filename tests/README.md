# Regression tests

A small pytest suite that checks pylmm against the Phase 1 golden outputs in
`tests/golden/` (see `tests/golden/README.md` for what each file holds). Its
job is to tell us whether a change, in particular the Python 3 port, changed
any numbers. It is not a unit-test suite.

## Running

```
docker/py27/test.sh                 # = docker/py27/run.sh pytest -q
docker/py27/test.sh --runslow       # include @pytest.mark.slow tests
docker/py27/test.sh -k cli          # any pytest arguments pass through
```

If you change the Dockerfile, rebuild first with `docker/py27/run.sh --build`.
The image pins pytest 4.6.11, the last release that supports Python 2.

The fast suite takes about 1 minute under emulated linux/amd64 on an Apple
Silicon Mac. The two slow tests run the README commands on the full BED data
(kinship and GWAS) and add about 3 minutes.

The tests are written to run unchanged on Python 3. Today they fail on
Python 3 only because pylmm itself is Python 2 code.

## Layout

| file | covers |
|---|---|
| `helpers.py` | paths, **all tolerances**, golden readers, subset writers, CLI runner, the one npdump loader |
| `conftest.py` | `--runslow`, session data fixtures (`hmdp`, `bed_null`) |
| `test_kinship.py` | `calculateKinship` (hmdp, centered, BED), `pylmmKinship.py` BED and EMMA paths, eigenvalues, symmetry/PSD, npdump = text |
| `test_null_fit.py` | `LMM.fit` REML/ML, LL grid, missing Y, covariate, BED h=0 boundary, h in [0,1] |
| `test_association.py` | `lmm.GWAS` REML/ML × refit with missing genotypes, `association(returnBeta=True)` on imputed SNPs, p-value vs t-stat/df |
| `test_misc.py` | `misc.fitTwo` |
| `test_cli_gwas.py` | `pylmmGWAS.py` BED (default, `--REML`, `--refit`, `--REML --refit`, `--removeMissingGenotypes`) and EMMA (with/without `--emmaCOV`), run in `tmp_path` |
| `test_example_script.py` | `scripts/example.py` end to end, via `runpy` |

CLI outputs are parsed as numbers, never compared as text, because `str()`
precision depends on the numpy version.

## Tolerances

All of them are defined in `helpers.py`.

- `RTOL=1e-10, ATOL=1e-12`: deterministic linear algebra with no optimiser
  involved (kinships, the LL grid at fixed h).
- `RTOL_FIT=1e-5, ATOL_FIT=1e-8`: anything downstream of
  `scipy.optimize.brent` (fitted h, sigma, beta, LL, and every t-stat or
  p-value). Brent stops at about 1.5e-8 relative in h. Perturbing the hmdp
  kinship by 1e-14 relative moved h by 5e-8, t-stats by up to 5.6e-6 relative
  (about 2e-8 absolute for |t| near 3e-3), and p-values by up to 8e-7. A
  different BLAS or numpy will cause perturbations of that size.
- Eigenvalue files: `atol = 1e-10 * max(eig)`. Most entries are LAPACK
  round-off near 0, so their sign and relative error are meaningless.
- Invariants: symmetry `atol 1e-12`; PSD means `min eig >= -1e-10 * max eig`;
  p must equal `2*t.sf(|t|, df)` to `rtol 1e-10`.
- nan positions must match exactly in every comparison.

Today, on Python 2.7, every value matches bit for bit. The looser tiers are
headroom for Phase 3. If a Phase 3 failure falls just outside them, check
whether the cause is an optimiser path difference before loosening anything,
and record the reason here.

## Known gaps (by design)

- `ts/ps_ml_norefit` in `assoc_hmdp_GWAS.txt` are identical to the REML
  columns, because of known bugs (`docs/phase1-baseline.md` #1 and #9). That
  test therefore does not exercise ML. ML fitting is covered directly by
  `test_null_fit.py` (`hmdp_ml`, `bed_ml`).
- REML refit through `lmm.GWAS` is checked on the first 200 SNPs only. All
  1000 are covered through `example.py`.
- `--removeMissingGenotypes` gets no p-value/df check, because its per-SNP N
  isn't in the output.
- The only test that reads `data/*.npdump` pickles goes through
  `helpers.load_npdump`. `scripts/example.py` also loads them itself.
