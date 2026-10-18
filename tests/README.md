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

The fast suite (32 tests) takes about 1 minute under emulated linux/amd64 on
an Apple Silicon Mac. The two slow tests run the README commands on the full
BED data (kinship and GWAS) and add about 2.5 minutes.

Every CLI run happens in the test's `tmp_path` and is killed after
`helpers.CLI_TIMEOUT` (900 s) by a `threading.Timer` watchdog, so a port bug
that makes a script loop forever fails the test instead of hanging CI. (The
watchdog was chosen over pytest-timeout because it needs no extra dependency
on either Python.)

The same tests run unchanged on Python 3.10+, which is what CI uses:

```
uv run pytest                       # fast suite: 32 passed, 2 skipped
uv run pytest --runslow             # include the slow tests
just test -k "kinship or null"      # any pytest arguments pass through
```

The py2.7 image gives the same result on the ported code.

## Layout

| file | covers |
|---|---|
| `helpers.py` | paths, **all tolerances**, golden readers, input writers (BED/TPED/EMMA subsets, covariate and missing-phenotype files, header-less and gzipped kinship), CLI runner with timeout and `#`-header guard, the one npdump loader |
| `conftest.py` | `--runslow`, session data fixtures (`hmdp`, `bed_null`) |
| `test_kinship.py` | `calculateKinship` (hmdp, centered, BED), `pylmmKinship.py` BED and EMMA paths, eigenvalues, symmetry/PSD, npdump = text. 1219×1219 kinships are checked through the leading and trailing 100×100 blocks, the diagonal and `K.dot(cos(arange(n)))` |
| `test_null_fit.py` | `LMM.fit` REML/ML, LL grid, missing Y, covariate, BED h=0 boundary, h in [0,1] |
| `test_association.py` | `lmm.GWAS` REML/ML × refit with missing genotypes, `association(returnBeta=True)` on imputed SNPs, p-value vs t-stat/df |
| `test_misc.py` | `misc.fitTwo` |
| `test_cli_gwas.py` | `pylmmGWAS.py`: BED (default, `--REML`, `--refit`, `--REML --refit`, `--removeMissingGenotypes`, `--covfile`, `NA`/`-9` phenotypes), `--tfile`, EMMA (with/without `--emmaCOV`, gzipped `--kfile`, `NA` phenotypes). Each run also checks the null fit printed by `-v` against `gwas_cli_nullfit.kv.txt` and, where the same model exists, against the API fit in `null_fits.kv.txt`; the EMMA t-stats are also checked against the API's `assoc_hmdp_imputed.txt` |
| `test_example_script.py` | `scripts/example.py` end to end, via `runpy`; `LMM.fit`/`association` are wrapped to record the intermediate null fit and first association |

CLI outputs are parsed as numbers, never compared as text, because `str()`
precision depends on the numpy version.

## Tolerances

All of them are defined in `helpers.py`, with the measurements behind them.

- `RTOL=1e-10, ATOL=1e-12`: deterministic linear algebra with no optimiser
  involved (kinships, the LL grid at fixed h).
- `RTOL_FIT, ATOL_FIT`: anything downstream of `scipy.optimize.brent`
  (fitted h, sigma, beta, LL, t-statistics). `RTOL_FIT=1e-5`, `ATOL_FIT=5e-6`.
  Measured by perturbing the hmdp kinship as `K*(1 + eps*N)` (N symmetric
  standard normal; 200 seeds at eps=1e-15 and 100 at 1e-14) and rerunning
  the null fit, `lmm.GWAS` (with and without refit) and the imputed
  association. The worst changes were 5.7e-8 in h, 5.0e-7 absolute in a
  t-stat (a per-SNP refit; up to 1.6e-5 relative for |t| near 1e-3) and
  1.1e-8 in beta. `ATOL_FIT` is 10x the worst absolute t-stat change. The
  earlier `ATOL_FIT=1e-8` was too tight: a 1e-15 perturbation could exceed
  it. The measurement used the hmdp data only; the BED CLI runs share the
  same code paths.
- `RTOL_P` with `atol=0`: p-values, compared purely relatively so that tiny
  p-values are checked too (an absolute tolerance of 1e-8 would accept any
  p below 1e-8). `RTOL_P=1e-5`; the worst measured relative p-value change
  in the same experiment was 1.2e-6.
- `CLI_PRINT_ATOL` (1e-3): the `heritability=%0.3f, sigma=%0.3f` line of
  `pylmmGWAS.py -v`, one unit in the last printed digit. Against the API fit
  the printed value must be within rounding (5e-4).
- Eigenvalue files: `atol = 1e-10 * max(eig)`. Most entries are LAPACK
  round-off near 0, so their sign and relative error are meaningless.
- Invariants: symmetry `atol 1e-12`; PSD means `min eig >= -1e-10 * max eig`;
  p must equal `2*t.sf(|t|, df)` to `rtol 1e-10`.
- nan positions must match exactly in every comparison.

Today, on Python 2.7, every value matches bit for bit. The looser tiers are
headroom for Phase 3. If a Phase 3 failure falls just outside them, check
whether the cause is an optimiser path difference before loosening anything,
and record the reason here.

## CLI inputs must be header-less

`pylmmGWAS.py` reads `--kfile` with `np.fromfile(sep=' ')`, which stops at
the first `#`, and then `K.resize()` zero-fills the rest without a warning
(`docs/phase1-baseline.md` bug 16). `tests/golden/hmdp_K_npdump.txt` has `#`
header lines, so the EMMA tests write a header-less copy
(`helpers.write_headerless_kinship`). `helpers.run_script` (and the
generator's `run_cli`) refuse any input file that starts with `#`.

## Known gaps (by design)

- `ts/ps_ml_norefit` in `assoc_hmdp_GWAS.txt` are identical to the REML
  columns, because of known bugs (`docs/phase1-baseline.md` #1 and #9). That
  test therefore does not exercise ML. ML fitting is covered directly by
  `test_null_fit.py` (`hmdp_ml`, `bed_ml`).
- REML refit through `lmm.GWAS` is checked on the first 200 SNPs only. All
  1000 are covered through `example.py`.
- `--removeMissingGenotypes` gets no p-value/df check, because its per-SNP N
  isn't in the output.
- The null fit printed by `pylmmGWAS.py -v` has only 3 decimals, and the BED
  runs fit at the h=0 boundary, so for BED the printed h cannot tell a good
  kinship from a bad one; the per-SNP goldens do. With `--refit` nothing is
  printed.
- Not covered: `--kfile2` (broken, bug 6), `--eigen` (cannot read what
  pylmmKinship writes, bug 7), `--noMean`, `-p` (phenotype index > 0),
  `--covfile` with missing values (the CLI refuses them), the fastLMM-style
  kinship reader in `input.readKinship` (unreachable from the CLIs),
  `pylmmKinship.py --tfile`, and `.kin` auto-discovery next to the genotype
  files.
- The TPED and covariate inputs are synthetic (derived from the BED subset);
  TPED uses only allele codes `1`/`2`/`0`, the only ones
  `input.getGenos_tped` understands.
- The only test that reads `data/*.npdump` pickles goes through
  `helpers.load_npdump`. `scripts/example.py` also loads them itself.
