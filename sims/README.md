# sims/: simulation framework

This directory holds simulation studies that check whether pylmm gives the
right answers, and that measure the suspected bugs listed in
`docs/phase1-baseline.md`. The studies measure those bugs; they do not fix
them. `sims/` lives at the top level of the repo and is not part of the
installed package. Its extra dependencies (pandas, matplotlib, statsmodels)
are in the `sims` uv dependency group:

```
uv sync --group sims                       # or: just sync
just sim example_null --quick              # uv run --group sims python -m sims run example_null --quick
just sim example_null --jobs 6             # full run
just sim-report example_null               # summary.csv / summary.md / PNGs
uv run --group sims python -m sims plan example_null   # list cells, count tasks
uv run --group sims python -m sims list                # list studies
```

## Layout

| module | what it gives you |
|---|---|
| `genotypes` | `get(spec)` returns a cached, read-only `Genotypes(G, snp_ids, indiv_ids, pops, meta)`. Sources: `bed` (1219 mice, 98,740 SNPs, read with `pylmm.plink_io.PlinkReader`), `hmdp` (94 strains, about 132k SNPs, text), and seeded synthetic sources `unrelated`, `balding_nichols` (k populations with F_st, optional Dirichlet admixture) and `family` (nuclear sibships). `with_missing(G, rate, rng)` adds missing calls. |
| `kinship` | `get(geno_spec, snps=...)` returns a cached `Kinship(K, kva, kve)`. K is built by pylmm's own `calculate_kinship`, optionally on a separate SNP set. `split_snps(...)` splits SNPs into kinship and test sets. `from_matrix(K)` wraps any K. |
| `phenotypes` | `simulate(kin, h2, rng, G=, causal_idx=, causal_var=, X0=/n_covariates=, missing_rate=)` simulates y = X0·α + Σβ_j g_j + u + e. `choose_causal(...)` picks causal SNPs. |
| `methods` | Thin pylmm wrappers: `pylmm_null`, `pylmm_gwas` (that is, `lmm.gwas`), `pylmm_association`. Also `ols_gwas`, an OLS baseline. |
| `oracle` | Independent dense REML/ML fit, for small n (see below). |
| `metrics` | `lambda_gc`, `type1_error`, `power`, `qq_data`, `rate_ci`, `bias`/`rmse`/`coverage`/`estimate_summary`, `compare` (agreement between two sets of estimates, such as pylmm and the oracle), `pool_pvalues`. |
| `plots` | `qq`, `lines`, `scatter`. Each writes one PNG. |
| `runner` | Grid × replicates. Each task gets its own deterministic seed. Results are appended to JSONL, runs can resume, and tasks can run in parallel. |
| `report` | `load_results`, `summarize`, `write_summary`, `markdown_table`. |
| `bench/` | Phase 6 speed benchmarks, one module each, run as `python -m sims.bench.<name> [--quick]` (seeded; each prints a markdown table and checks that the fast path gives the same numbers). `bed_blocks`: `PlinkReader.iter_snps` vs `iter_blocks`. |

## Writing a study

Copy `studies/example_null.py` to `studies/<name>.py`, for example
`s1_null_calibration.py`. A study is a module with these attributes:

```python
NAME = "s1_null_calibration"
GRID = {"n": [200, 500, 1000], "h2": [0.0, 0.5, 0.9], "geno_seed": [1]}   # cartesian product
QUICK_GRID = {"n": [200], "h2": [0.5], "geno_seed": [1]}                  # tiny smoke grid
REPS, QUICK_REPS = 50, 2
SEED_PARAMS = None   # optional: ["n", "h2"] makes cells that differ only in
                     # other params (e.g. method) share seeds (common random numbers)
STUDY_VERSION = 1    # optional; bump when run_cell's output changes meaning

def run_cell(params: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ...                      # simulate, fit, measure
    return {"lambda_lmm": ..., "h_hat": ..., "qq_p": small_list}

def report(df, outdir):      # optional; the default is a mean/SE table per cell
    ...
```

- `GRID` can also be a list of param dicts, which suits grids that are not
  cartesian. A study can instead define `grid(quick)` and `reps(quick)`.
- Params must be JSON-able: numbers, strings, booleans, small lists. The cell
  id is a hash of the params, so changing a param value makes a new cell.
  Renaming a param also makes new cells, so don't rename params once a full
  run exists.
- If you change what `run_cell` computes for the same params (a metric's
  definition, a simulation fix), bump `STUDY_VERSION`. Every row records its
  version, and from version 2 on the version is part of the cell id, so the
  next run recomputes every cell and the loaders (`runner.current_rows`,
  `report.load_results`, `synthesis.py`) ignore the old rows. They also
  ignore rows of cells that are no longer in the grid. Rows written before
  `STUDY_VERSION` existed count as version 1.
- `run_cell` must be a pure function of `(params, seed)`. Draw all randomness
  from `np.random.default_rng(seed)`. Keep the genotype design fixed per cell
  with a `geno_seed` param passed to `genotypes.get`. That also lets the
  per-worker cache reuse genotypes and the eigendecomposition of K across
  replicates. Metric names must not clash with param names.
- Return scalars plus at most a few short lists, at most a few hundred numbers
  per replicate. For example, store a random sample of 200 null p-values for
  a pooled QQ plot, not the full scan.
- Catch only the pylmm failures you want to measure (S5), and turn them into
  metrics such as `crashed: 1`. Any other exception is written to
  `raw/<mode>.errors.jsonl` and retried on the next run. The run then exits
  with a non-zero status.
- To get the same random data for several methods, compute all the methods
  inside one `run_cell`. Alternatively, put the method in the grid and use
  `SEED_PARAMS`.
- Use the kinship pylmm itself would use, `kinship.get(spec, snps=...)`. Pass
  the eigendecomposition through the wrappers in `methods`, which copy `kva`
  for you.

## Quick vs full, and compute budget

- `--quick` is a smoke test. It must finish in under about 3 min on one core,
  and ideally within seconds. CI runs `example_null --quick`.
- Full runs: the runner caps `--jobs` at the number of CPUs minus 2 (at
  least 1; 6 on the 8-core, 16 GB machine the benchmarks ran on) unless you
  pass `--no-job-cap`. Workers are spawned with `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`.
  Keep each worker under about 1.5 GB:
  - A 1219×1219 kinship plus its eigenvectors is about 24 MB.
  - A full BED matrix is about 960 MB, so subset SNPs with
    `{"source": "bed", "n_snps": ...}`.
  - Each worker caches at most 8 genotype matrices and 4 kinships.
- Aim for a full run of under about 30 min with `--jobs 6`. To estimate it,
  time a quick run and scale up. `raw/runs.jsonl` records wall time and the
  sum of task times for each invocation. `summary.md` repeats them.
- The oracle costs O(n³) per likelihood evaluation, with about 220
  evaluations per fit. That is about 0.1 s at n = 200 and about 0.6 s at
  n = 500. Use it for n ≤ 500, and on a subset of replicates or SNPs.
- Runs are resumable. If you interrupt a run and start it again, finished
  (cell, rep) pairs are skipped, so a rerun in place computes only the
  missing replicates. Failed replicates are never written to
  `raw/<mode>.jsonl` (only to `<mode>.errors.jsonl`, rewritten on each run),
  so they are always among the missing ones and are retried. `--cells ID …`
  restricts a run to some cells (ids from `python -m sims plan <study>`).
  Running with a larger `--reps` adds only the new replicates. `--force`
  discards that mode's raw results.

## Where results go, and what to commit

```
sims/results/<study>/raw/full.jsonl       raw replicates (gitignored)
sims/results/<study>/raw/quick.jsonl      (gitignored)
sims/results/<study>/raw/runs.jsonl       invocation log (gitignored)
sims/results/<study>/summary.csv          COMMIT: one row per cell, means and SEs
sims/results/<study>/summary.md           COMMIT: the same table plus notes and run info
sims/results/<study>/*.png                COMMIT: a few figures (about 30 KB each)
sims/results/<study>/quick/               report of a quick run (gitignored)
```

Commit the study code and the full-run summaries and figures. Never commit
`raw/`. `.gitignore` covers `sims/results/**/raw/` and `sims/results/**/quick/`.
If you need a result in a write-up, put it in `summary.csv`. Set
`SIMS_RESULTS_DIR` to redirect all results; the tests do this.

## The oracle (`sims/oracle.py`)

The oracle is a deliberately simple, independent fit of
y = Xβ + u + e with V = σ²(h·K + (1−h)·I), the same parameterization pylmm
uses. It does not reuse any pylmm code:

- It factors the dense H(h) with Cholesky at each h and uses the textbook
  profile ML/REML formulas. The REML formula includes the +½log|X'X| term, so
  it equals the likelihood of the error contrasts.
- It evaluates a 201-point grid over [0, 1] (endpoints included), refines
  every interior local maximum with bounded Brent, and returns the best
  point.
- `wald(y, K, x, X0, null_reml=True, test_reml=None, h=None, refit=False,
  sigma_convention="textbook")` returns the Wald t and p. The p-value uses
  the t distribution with df = n−p, as pylmm does.
- `fit` reports both `sigma2_ml` (Q/n) and `sigma2_reml` (Q/(n−p)).
  `sigma2` is whichever one matches the fitted likelihood, and
  `sigma2_kind` says which.
- Switches for pylmm's conventions default to the textbook behaviour. The
  section "Isolating pylmm's suspected bugs" below lists them.

`tests/sims/test_sims_oracle.py` validates it:

- K = I gives a flat profile (`identifiable=False`, h = 0), with OLS β, σ²
  and t.
- Balanced one-way ANOVA matches the closed-form REML and ML estimators.
- The REML LL equals the error-contrast log-density, and the ML LL equals the
  `multivariate_normal` log-density.
- The global maximum is checked against a 2001-point grid.
- statsmodels `MixedLM` agrees on unbalanced groups, for REML and ML, to
  1e-5 in h. statsmodels' REML llf omits the constant ½log|X'X|.

Assumptions and known behaviour:

- y and X must be complete, with X of full rank. K must be symmetric PSD.
- **ML with pylmm's GRM can be unbounded.** `calculate_kinship` centers each
  SNP, so K·1 = 0. When K is built from more SNPs than individuals (outbred
  synthetic data), its null space is exactly span(1). With an intercept, the
  ML profile then goes to +∞ as h→1, so the MLE does not exist. Kinships with
  a larger null space, such as the BED data (rank about 110), do not have
  this problem. In that case `fit` returns h just below 1 and sets
  `ml_unbounded=True`. REML is unaffected. pylmm hides this because it clamps
  eigenvalues to at least 1e-6 and its grid stops at 0.99. This matters for
  S2 (ML with h near 1).
- pylmm's ML LL is the standard profile ML LL, so raw ML values match the
  oracle's `ll`. pylmm's REML LL is the standard REML LL plus a constant,
  `pylmm_reml_offset(n, p)` = n/2·log(n/(n−p)) − p/2. Both depend on Q only
  through −(n−p)/2·log Q, so the REML ĥ is the same. Never compare raw REML
  LLs. Compare ĥ, β̂, t and LL differences instead, or use
  `OracleFit.ll_pylmm`. A test checks the offset at 8 values of h, from 0 to
  0.999.

## pylmm behaviour the framework works around (not fixed)

- `LMM.__init__` clamps the caller's `kva` in place (bug 12), and
  `calculate_kinship` mean-imputes and standardizes its argument in place. The
  framework always passes copies: `Kinship.eig_for_pylmm()` and
  `kinship.kinship()`. Cached arrays are read-only, so any accidental
  mutation raises an error instead of silently corrupting later replicates.
- pylmm prints to stderr (for example "Read N individuals"). The wrappers in
  `methods` silence it.
- `lmm.gwas` never passes `REML` to the null fit (bug 1). `methods.pylmm_gwas`
  calls it as is, because that behaviour is what the studies measure.

Already seen in `example_null`: in 2 of 20 replicates at n = 200 and
h² = 0.5, the oracle's REML h is above 0.99 but pylmm returns 0.99 (bugs
2 and 3). In all other replicates pylmm matches the oracle to about 1e-8 in
h, and to 1e-12 in LL after the constant offset.

A one-off check on a real BED subset (300 random mice, a kinship from 2500
SNPs with rank 99, h² ∈ {0.2, 0.5, 0.8}) gave these results:

- The null-fit h matches the oracle to 1e-5 for both REML and ML.
- The LL matches to about 2e-6 after the offset. The small gap comes from
  pylmm's eigenvalue clamp.
- With REML, GWAS t-statistics agree to 1e-6.
- With `reml=False`, the GWAS t-statistics differ from the oracle's ML Wald
  t by up to 0.006. The cause is bug 1: the null h is still fitted by REML.

## Isolating pylmm's suspected bugs

Bug numbers refer to `docs/phase1-baseline.md`. Every switch below defaults
to the textbook behaviour. With all of them set to the pylmm setting, the
oracle reproduces pylmm. Flip one switch at a time to measure one effect on
its own.

| switch | textbook (default) | pylmm setting | isolates |
|---|---|---|---|
| `fit/wald(h_grid=)` | `"fine"`: global max on a 201-point grid over [0, 1], plus bounded Brent | `"pylmm"`: pylmm's exact search. Grid 0..0.99 in steps of 0.01, the `range(1, n−2)` scan, the first local optimum rather than the best, unbounded Brent with the h<0 guard, nan replaced by H[i−1], and the endpoint fallback | bugs **2** (off-by-one), **3** (grid excludes 1.0), **4** (unbounded Brent, nan fallback), and the first-not-best optimum |
| `fit/wald(eig_clamp=)` | `None` | `1e-6`: eigenvalues of K below 1e-6 are raised to 1e-6 (`LMM.__init__`) | the clamp itself. It matters as h → 1 and for rank-deficient K: the BED K has 1109 of 1219 eigenvalues below 1e-6. The in-place mutation of the caller's `kva` is bug **12** |
| `wald(null_reml=, test_reml=)` | `test_reml = null_reml` | `null_reml=True, test_reml=False` reproduces `gwas(reml=False)` | bug **1**: the null fit ignores `REML` |
| `wald(sigma_convention=)` and `OracleFit.sigma2_ml` / `sigma2_reml` | `"textbook"`: Q/n under ML | `"pylmm"`: Q/(n−q) always. pylmm's reported sigma is `sigma2_reml` | bug **9**: sigma is not the ML estimate under ML |
| `OracleFit.ll` / `ll_pylmm` | standard (RE)ML | `ll_pylmm` adds the REML constant | a convention, not a bug. Use it to compare raw LLs |

Usage:

```python
PYLMM = dict(eig_clamp=1e-6, h_grid="pylmm")
f_pylmm = oracle.fit(y, K, X0, reml=True, **PYLMM)       # reproduces pylmm
f_true = oracle.fit(y, K, X0, reml=True)                 # textbook
f_opt = oracle.fit(y, K, X0, reml=True, eig_clamp=1e-6)  # differs from f_pylmm only by the optimizer
w = oracle.wald(y, K, x, X0, null_reml=True, test_reml=False,
                sigma_convention="pylmm", **PYLMM)       # = lmm.gwas(reml=False)
```

Validation is in `tests/sims/test_sims_oracle.py`. With all switches set to
pylmm conventions, the oracle matches pylmm `LMM.fit` and `lmm.gwas` (REML,
and reml=False with a REML null) in three settings:

- synthetic rank-deficient K: h, LL (via `ll_pylmm`), β, sigma, t and p;
- a real BED subset (300 mice, rank about 99): |Δh| ≤ 3e-8, |ΔLL| ≤ 1e-13,
  |Δt| ≤ 1e-8;
- the two `example_null` replicates where pylmm stopped at 0.99: h = 0.99
  exactly, and |Δt| ≈ 1e-13.

In those two `example_null` replicates the textbook oracle gives h of 1.0
and 0.996. That is still true with `eig_clamp=1e-6`, so the optimizer is the
cause (bugs 2 and 3), not the clamp. Their textbook t-statistics differ from
pylmm by 0.003 to 0.016.

Not covered by oracle switches:

- Bug 8 (kinship normalization) is a kinship-construction difference.
  Compare `kinship.kinship` (pylmm's `calculate_kinship`) with a K built from
  `pylmm.plink_io.PlinkReader.normalize_genotype` directly.
- The `misc.fit_two` grid (bug 3) involves two kinships, and the oracle fits
  only one.
- Bug 5 (`tstat(log=True)`) is on an unused code path.
