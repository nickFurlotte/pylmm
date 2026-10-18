# S1: null calibration of pylmm vs OLS

Question: with no causal SNPs, are pylmm's p-values uniform across population
structures and heritabilities, and how does that compare with OLS?

Code: `sims/studies/s1_null_calibration.py`. Run with
`just sim s1_null_calibration --jobs 6`, then `just sim-report s1_null_calibration`.

## Design

Each cell is one (population structure, h2) pair, n = 600 individuals, and a
fixed genotype design (`geno_seed = 1`). In every replicate we draw a new
phenotype y = mu + u + e, with u ~ N(0, h2 K) and no SNP effects, so every
tested SNP is null.

| factor | levels |
|---|---|
| structure (`pop`) | `unrelated` (independent HWE SNPs); `bn_fst0.05`, `bn_fst0.2` (Balding-Nichols, 3 populations); `family` (150 nuclear sibships of 4); `bed` (600 random mice from the bundled BED data, 13,500 random SNPs) |
| h2 | 0, 0.2, 0.5, 0.8, 0.95 |
| K SNPs | 6000 SNPs, K from `pylmm.lmm.calculateKinship` |
| tested SNPs | 5000 SNPs per test set, MAF >= 0.05, missing calls mean-imputed (the CLI default) |
| replicates | 80 per cell (25 cells) |

K is also the true covariance used to simulate u. So when the tested SNPs
are disjoint from K, the model is exactly right for the unrelated and
family designs; the BED and Balding-Nichols cases carry extra structure
that K (6000 SNPs) only estimates.

Two test sets per replicate:

* **disjoint**: 5000 SNPs not used in K (no proximal contamination);
* **in K** (suffix `_in`): 5000 SNPs taken from the 6000 K SNPs (the usual
  "K from all SNPs" practice, which gives proximal contamination).

Methods on each test set:

* `ols`: per-SNP OLS t-test, no kinship (`methods.ols_gwas`).
* `reml`: `pylmm.lmm.GWAS(REML=True)`.
* `cli`: `GWAS(REML=False)`, which is what `pylmmGWAS` does by default (no
  `--REML`, no `--refit`). `GWAS` and the CLI never pass `REML` to the null
  fit (bug 1), so both `reml` and `cli` fit h by REML under the null and only
  the per-SNP Wald step differs.

Metrics: lambda_GC with a 95% CI (mean +/- 1.96 SE over replicates); type-I
error at alpha = 0.05, 1e-2, 1e-3, 1e-4, pooled over all replicates and SNPs
with Wilson CIs (reported as observed/alpha, with `*` when the CI excludes
1); the number of hits at 1e-4; the mean REML h-hat; and QQ plots built from
stored p-value tails and random samples.

Outputs in `sims/results/s1_null_calibration/`: `summary.csv` (all numbers),
`summary.md`, `qq_panel.png` (rows = structure, columns = h2 of 0, 0.5, 0.95),
`lambda_vs_h2.png` (lambda_GC against h2 per structure).

## Hypotheses

1. **OLS** is calibrated for `unrelated` and badly inflated wherever there is
   structure: `bn_fst0.2`, `bed` (lambda_GC in the tens, since the BED kinship
   has effective rank about 110), and `family` (milder, growing with h2). It is
   also calibrated whenever h2 = 0, whatever the structure.
2. **pylmm, tested SNPs disjoint from K**: lambda_GC and type-I error are about
   1 at all h2 for `unrelated`, `family` and `bn_*`. For `bed` expect some
   inflation, because 6000 random SNPs do not capture all LD and replicate
   structure. Any inflation at h2 near 0.95 or in the tails at 1e-4 would point
   at the h clamp at 0.99 (bugs 2 and 3: the fitted h cannot exceed 0.99).
3. **pylmm, tested SNPs in K**: deflation (lambda_GC < 1, observed/alpha < 1),
   strongest when K has low rank (`bed`) and at high h2, where the random
   effect absorbs the SNP.
4. **`cli` vs `reml`**: in this code the `REML` flag passed to `association`
   changes only the log-likelihood value, not the Wald t, so the two should
   agree to rounding. The quick run already shows identical results. They are
   both run so that the study documents this; any difference would be a new
   finding.

## How to read the results

* lambda_GC near 1 with a CI containing 1: calibrated. The CI is over
  replicates, so it includes the replicate-to-replicate variation from
  sharing y across SNPs. This is why we have many phenotypes and not one
  large scan.
* Observed/alpha well above 1 at 1e-3 or 1e-4 with lambda_GC near 1 means
  heavy tails with a calibrated bulk (typical of a mis-estimated h in a few
  replicates). Use `hits_1e-04` and `t1e_1e-04_lo/hi` in `summary.csv`.
* Wilson CIs treat SNPs as independent. For `bed` (LD) and `family` this is
  optimistic, so treat a `*` at alpha <= 1e-3 there as suggestive, and look
  at the QQ plot and the replicate-level lambda_GC.
* With 4e5 tests per cell, 1e-4 has about 40 expected hits, so the Wilson CI
  is about +/- 30% of alpha. Ratios between 0.7 and 1.3 are indistinguishable
  from calibrated.
* Gap between `reml` and `reml_in` = deflation from proximal contamination.
  Gap between `ols` and `reml` = what the kinship correction buys.

## Compute

One replicate takes about 2.4 to 2.8 s single-core at n = 600 with 5000 SNPs
per test set (three scans per set, each about 0.4 s of pylmm and the rest
in OLS and p-value bookkeeping). The full grid is 25 cells x 80 reps = 2000 tasks,
about 5200 core-seconds. At `--jobs 6` that is about 15 min if cores scale
perfectly; allow about 20 to 25 min on the shared laptop. Memory per worker
is under 300 MB. `--reps` can be raised on top of an existing run
(only new replicates run) if the 1e-4 CIs are too wide.

The quick grid (3 cells: `unrelated`, `family`, `bed` at n = 200, 300 SNPs,
2 replicates) takes about 2.5 s of wall time plus the Python start-up.

## Framework gaps handled locally

* Missing calls are mean-imputed per SNP and low-MAF SNPs dropped in the
  study module, because `GWAS` otherwise refits an eigendecomposition per SNP
  with missing data.
* The QQ plot pools stored p-value tails (exact) and a random sample (bulk)
  in the study module, because `metrics.qq_data` expects the full p-value
  vector, which is too large to store per replicate.
