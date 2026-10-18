# sims/: Phase 4 simulation framework

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
| `genotypes` | `get(spec)` returns a cached, read-only `Genotypes(G, snp_ids, indiv_ids, pops, meta)`. Sources: `bed` (1219 mice, 98,740 SNPs, read with `pylmm.input.plink`), `hmdp` (94 strains, about 132k SNPs, text), and seeded synthetic sources `unrelated`, `balding_nichols` (k populations with F_st, optional Dirichlet admixture) and `family` (nuclear sibships). `with_missing(G, rate, rng)` adds missing calls. |
| `kinship` | `get(geno_spec, snps=...)` returns a cached `Kinship(K, Kva, Kve)`. K is built by pylmm's own `calculateKinship`, optionally on a separate SNP set. `split_snps(...)` splits SNPs into kinship and test sets. `from_matrix(K)` wraps any K. |
| `phenotypes` | `simulate(kin, h2, rng, G=, causal_idx=, causal_var=, X0=/n_covariates=, missing_rate=)` simulates y = X0·α + Σβ_j g_j + u + e. `choose_causal(...)` picks causal SNPs. |
| `methods` | Thin pylmm wrappers: `pylmm_null`, `pylmm_gwas` (that is, `lmm.GWAS`), `pylmm_association`. Also `ols_gwas`, an OLS baseline. |
| `oracle` | Independent dense REML/ML fit, for small n (see below). |
| `metrics` | `lambda_gc`, `type1_error`, `power`, `qq_data`, `rate_ci`, `bias`/`rmse`/`coverage`/`estimate_summary`, `compare` (agreement between two sets of estimates, such as pylmm and the oracle), `pool_pvalues`. |
| `plots` | `qq`, `lines`, `scatter`. Each writes one PNG. |
| `runner` | Grid × replicates. Each task gets its own deterministic seed. Results are appended to JSONL, runs can resume, and tasks can run in parallel. |
| `report` | `load_results`, `summarize`, `write_summary`, `markdown_table`. |

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
  the eigendecomposition through the wrappers in `methods`, which copy `Kva`
  for you.

## Quick vs full, and compute budget

- `--quick` is a smoke test. It must finish in under about 3 min on one core,
  and ideally within seconds. CI runs `example_null --quick`.
- Full runs: the laptop has 8 cores (4 performance) and 16 GB of RAM. Use
  `--jobs 6` at most; the runner caps `--jobs` at 6 unless you pass
  `--no-job-cap`. Workers are spawned with `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`.
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
  (cell, rep) pairs are skipped. Running with a larger `--reps` adds only the
  new replicates. `--force` discards that mode's raw results.

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
- `wald(y, K, x, X0, h=None, refit=False)` returns the Wald t and p.
  σ² = Q/(n−p) and df = n−p, which is pylmm's convention.

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
- **ML with pylmm's GRM can be unbounded.** `calculateKinship` centers each
  SNP, so K·1 = 0. When K is built from more SNPs than individuals (outbred
  synthetic data), its null space is exactly span(1). With an intercept, the
  ML profile then goes to +∞ as h→1, so the MLE does not exist. Kinships with
  a larger null space, such as the BED data (rank about 110), do not have
  this problem. In that case `fit` returns h just below 1 and sets
  `ml_unbounded=True`. REML is unaffected. pylmm hides this because it clamps
  eigenvalues to at least 1e-6 and its grid stops at 0.99. This matters for
  S2 (ML with h near 1).
- pylmm's REML LL equals the oracle's REML LL plus `pylmm_reml_offset(n, p)`,
  a constant in h. pylmm's ML LL equals the oracle's ML LL.

## pylmm behaviour the framework works around (not fixed)

- `LMM.__init__` clamps the caller's `Kva` in place (bug 12), and
  `calculateKinship` mean-imputes and standardizes its argument in place. The
  framework always passes copies: `Kinship.eig_for_pylmm()` and
  `kinship.kinship()`. Cached arrays are read-only, so any accidental
  mutation raises an error instead of silently corrupting later replicates.
- pylmm prints to stderr (for example "Read N individuals"). The wrappers in
  `methods` silence it.
- `lmm.GWAS` never passes `REML` to the null fit (bug 1). `methods.pylmm_gwas`
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
- With `REML=False`, the GWAS t-statistics differ from the oracle's ML Wald
  t by up to 0.006. The cause is bug 1: the null h is still fitted by REML.
