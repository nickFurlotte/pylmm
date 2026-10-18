# S2: heritability recovery (h2, sigma^2, beta)

Question: does pylmm's null fit recover h2 (and sigma^2, beta) accurately under
REML and ML, across genotype structures? Where it does not, which mechanism is
responsible: the optimizer, the eigenvalue clamp, or estimator bias that any
correct method has?

Code: `sims/studies/s2_h2_recovery.py`. Run with
`just sim s2_h2_recovery --jobs 6`, then `just sim-report s2_h2_recovery`.
Status: the framework and quick run are done; **the full grid has not been run
yet**, so Section 5 only has smoke-test observations. Fill in Section 5 after
the full run.

## 1. Design

Per replicate: y = 1*mu + u + e, mu = 1, u ~ N(0, h2 K), e ~ N(0, 1-h2), so the
true sigma^2 is 1 (up to the realised scale of K; pylmm's K has mean diagonal
about 1) and the true beta is 1. The kinship K is fixed per cell and is built
by pylmm's own `calculateKinship` from the same SNPs that generated the
genotypes, so the simulated u is exactly the model's u (no misspecification in
the main grid). The only randomness within a cell is (u, e).

| factor | levels |
|---|---|
| genotype source | `unrelated`; `balding_nichols` (k = 3, F_st = 0.1); `family` (nuclear sibships of 4; K is the sampled GRM, not the pedigree); `bed` (random subset of the real 1219-mouse data, SNPs and mice drawn at random) |
| n | 200, 600 |
| kinship SNPs m_K | "lo" = n/2 (m < n, K rank-deficient with rank ~ m - 1) and "hi" = 5n (m > n) |
| true h2 | 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 1.0 |
| genotype draw | `geno_seed` in {1, 2} (two K matrices per structure) |
| replicates | 12 per cell (24 per structure x n x m_K x h2) |

160 main cells x 12 reps = 1920 replicates, plus 12 probe cells (Section 4).

Note on `bed`: the real kinship has effective rank of about 110 regardless of
m_K (replicate inbred animals), so "lo" and "hi" differ only in noise; the
m > n null-space effect does not apply to it. Treat the two bed m_K levels as
near-replicates.

Methods fitted on every replicate:

* pylmm REML and pylmm ML (`methods.pylmm_null`).
* Oracle (`sims/oracle.py`, independent dense Cholesky fit) with textbook
  settings (T), and with pylmm's conventions switched on one at a time:
  G = `h_grid="pylmm"` only (optimizer: bugs 2, 3, 4 and the first-optimum
  rule), C = `eig_clamp=1e-6` only (clamp), B = both (reproduces pylmm; the
  maximum |h_B - h_pylmm| is reported as `max_abs_B_vs_pylmm` as a check).
  Each is done for REML and for ML.
* pylmm's own LL on a 201-point grid over [0, 1] (all replicates, cheap), to
  count profile modes and to measure the LL deficit of pylmm's answer.

The oracle is O(n^3) per likelihood evaluation: 8 fits per replicate. At
n = 200 it runs on every replicate. At n = 600 it runs on the replicates with
`seed % 3 == 0` (about 4 of 12), which is a deterministic and effectively
random subset. pylmm's own fit, boundary hits, mode counts and LL deficits are
available on all replicates; only the attribution table (and oracle columns)
use the oracle subset.

## 2. Hypotheses

* H1 (REML, interior h2): pylmm's REML h-hat matches the textbook oracle to
  about 1e-6 (the optimizer is fine when the maximum is in 0.02 .. 0.97).
  Remaining bias is sampling bias of REML itself, which is small except where
  the profile is flat (m_K = 5n at small n: K is nearly I plus noise, so h2 is
  weakly identified; boundary pile-ups at 0 and 1 cause bias towards the
  middle).
* H2 (h2 near 1): pylmm's grid stops at 0.99 and `getMax` ignores grid index
  n-2, so for true h2 >= 0.98 pylmm is biased low and pile-ups appear at exactly
  0.99 (or at 0 if the profile is increasing and the interior-peak test
  fails). The oracle with `h_grid="pylmm"` should reproduce this; the clamp
  should explain little of it. Attribution: "opt".
* H3 (ML with m_K > n, outbred): K has a null space equal to span(1), the
  intercept absorbs it, and the ML profile is unbounded as h -> 1. The oracle
  flags `ml_unbounded`. pylmm does not see the divergence (clamp +
  grid cap) but its answer then depends on the clamp and the grid:
  expect pylmm ML to pile up at 0.99 (or return a spurious interior mode) and
  to be much more biased than REML for any true h2, with the textbook oracle ML
  ĥ -> 1 in most replicates. Attribution: part "estimator" (the MLE
  does not exist), part "opt"/"clamp" (what pylmm does about it).
* H4 (m_K < n or real BED): K has a large null space with y not in it, so the
  ML profile is bounded and ML is only mildly biased downward relative to
  REML in the usual way (ML underestimates variance components by the
  p/n factor).
* H5 (sigma^2): under ML pylmm reports Q/(n-q), the REML scale, so its sigma is
  biased upward by n/(n-1) relative to the ML estimate. This is of order
  1/n (0.5% at n = 200), negligible against the h-hat effects, but exact and
  visible in the table (`sigma pylmm ML(reported)` vs `sigma2_ml implied`).
  sigma^2 and beta inherit errors from ĥ at the boundary (when ĥ is wrong,
  sigma is wrong).
* H6 (multiple optima): profiles with two interior maxima are rare when the
  model is correct. Misspecified phenotypes (the probe) should produce them,
  and then pylmm's first-optimum rule returns a lower-LL mode a measurable
  fraction of the time.

## 3. Metrics and tables (produced by `report()`)

All tables are written as CSV and into `summary.md`; figures are PNGs.

1. `bias_rmse`: per (structure, n, m_K, true h2): bias, RMSE and SD of ĥ for
   pylmm REML/ML and oracle REML/ML. SD is the sampling-distribution width; the
   plots show mean +- SD.
2. `h_recovery_<structure>.png`: ĥ vs true h2, panels n x m_K, pylmm (solid) vs
   oracle (dashed), REML blue, ML red.
3. `boundary`: fraction of replicates with pylmm ĥ = 0; pylmm ĥ at the 0.99
   cap; oracle ĥ = 1 (textbook); non-global optimum (pylmm's LL at ĥ below the
   best point on the 201-point grid of pylmm's own likelihood by more than
   1e-3 LL units); "lost-mode" (the same, and the profile has more than one
   mode, so it is a first-optimum failure and not a grid-edge failure);
   fraction multimodal; `ml_unbounded` frequency.
4. `attribution` (and `attribution_pooled_n`): decomposition of the mean pylmm
   bias, using the oracle variants on oracle replicates:
   `total = pylmm - truth = est + opt + clamp + inter` where
   est = T - truth (bias any correct estimator has), opt = G - T,
   clamp = C - T and inter = B - G - C + T. Means cancel across replicates, so
   `rms_opt`, `rms_clamp` and `rms_pylmm_vs_T` give the per-replicate size of the
   effects, and `rmse_T` vs `rmse_pylmm` show the accuracy cost. Caveat: where
   the ML MLE does not exist, T is the oracle's capped value (just below 1),
   which is a definition, not an estimate, so "est" for ML in that regime
   means "what the unbounded likelihood asks for".
5. `sigma_beta`: mean sigma (pylmm REML; pylmm ML as reported; the Q/n value
   implied; oracle sigma2_ml and sigma2_reml) and beta bias and SD.
6. `ml_cap`: ML vs REML bias/RMSE and 0.99-cap frequency by m_K, pooled over
   h2: this answers "how often does ML hit the cap and how much does it
   bias ML vs REML".
7. `probe`: multiple-optima probe.

## 4. Multiple-optima probe

12 cells, n = 100, K from Balding-Nichols (k = 3, 5n SNPs), 100 reps would be
cheap but the default of 12 is kept for uniformity (raise `--reps` for the probe
if needed; results are resumable). Phenotypes are deliberately
misspecified: `correct` (control), `mix_family` (extra family-structured
component, unit variance, not in K), `mix_bn` (extra component from a different
sibship draw), `mix_rank` (a few top eigenvectors given large extra variance).
Each is fitted with K alone, REML and ML. For all replicates (main grid and
probe) pylmm's own LL on [0, 1] is used to count modes (prominence >= 0.1 LL
units) and to compare the LL at pylmm's ĥ with the best grid point.

## 5. Observations so far

Smoke test only (`--quick`: 2 reps per cell, n = 200, 20 tasks, about 7 s with 2
jobs); not evidence, just checks the machinery.

* The all-switches oracle reproduces pylmm (`max_abs_B_vs_pylmm` = 0 to
  printed precision).
* With m_K = 5n (unrelated or family) ML had `ml_unbounded` in every replicate
  and the textbook oracle ML ĥ at 1; pylmm ML sat at 0.99 or at a
  interior mode, and the family case showed a large "clamp" effect on ML
  (the clamp changes the likelihood near h = 1, so pylmm's ML answer there
  is partly an artifact of the 1e-6 floor).
* At true h2 = 1 pylmm returns 0.99 (bias -0.01), of which the whole gap is
  the grid cap ("opt").
* Unrelated-genotype K from 1000 SNPs at n = 200 gives very noisy ĥ (RMSE about
  0.37 for REML at h2 = 0.5): this is estimator behaviour (K close to I), not a
  pylmm defect.

The full-grid interpretation is to be written after the run.

## 6. Compute estimate

Measured single-core (one process, BLAS threads = 1): an n = 200 replicate with
8 oracle fits is under 1 s (the quick run's 20 tasks took 13 CPU-s in total,
including genotype and kinship construction); an n = 600 replicate with the
oracle takes about 6 s, and one without it under 0.1 s (kinship and eigh are
cached per worker).

Full grid: 80 cells x 12 reps x ~0.7 s ~ 700 s at n = 200, and 80 cells x
(4 oracle reps x ~6 s + 8 x ~0.05 s) ~ 2000 s at n = 600, plus per-worker
genotype/kinship builds and BED reads (a few seconds each, up to 4 BED specs
per worker). Total ~3000 to 4000 CPU-s, so about 10 to 15 minutes at `--jobs 6`
(allow 20 for the 4 performance cores); the limit is 30. Memory per worker: the
largest K is 600 x 600 (3 MB) and the largest genotype matrix is 600 x 3000
(14 MB), so well under 1.5 GB. If it runs long, lower `ORACLE_MOD_600` fraction
(raise the modulus) rather than REPS.

## 7. Caveats

* The oracle's REML and ML maxima come from a 201-point grid plus Brent
  refinement; two modes closer than 0.005 in h would be missed (the pylmm LL
  grid used for mode counting has the same resolution).
* K is fixed per cell and only 2 geno_seeds per cell are used, so between-K
  variation is covered only coarsely; results for one structure are about
  these specific designs.
* h2 = 1.0 and 0 are degenerate (no e or no u); they are included on purpose as
  boundary cases.
* sigma^2 truth is 1 only up to the scale of K's diagonal (about 1).
