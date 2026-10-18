# S3: power of pylmm to detect causal SNPs

Code: `sims/studies/s3_power.py`. Run `just sim s3_power --quick` (about 6 s) or
`just sim s3_power --jobs 6` (full, see the estimate below). The full run has
not been executed yet; only the `--quick` smoke grid has.

## Questions

1. How much power does pylmm have to detect causal SNPs?
2. How does it compare with OLS, whose raw power is inflated by structure
   (false positives from confounding) and which needs calibrating before the
   comparison means anything?
3. Does per-SNP refit of h (`refit=True`) change power or effect-size accuracy
   compared with the fixed null-model h (`refit=False`)?
4. How much power is lost when the causal SNPs are in the kinship matrix
   (proximal contamination)?

## Design

Model: `y = sum_j beta_j g_j + u + e`, with `u ~ N(0, s_g K)`, `e ~ N(0, s_e I)`.
Each causal SNP explains `vpe` of the total variance (the generator rescales
beta by the realised SD of g). The rest is split `h2 : 1 - h2` between K and
noise. Signs are random, effects are equal across the causal SNPs of a cell.

| factor | levels |
|---|---|
| genotype source | `unrelated` (n=500), `balding_nichols` (k=3, F_st=0.1, n=500), `bed` (600 random mice, random SNPs) |
| background h2 | 0.2, 0.5, 0.8 |
| number of causal SNPs | 1, 5 |
| variance explained per causal SNP | 0.5%, 1%, 2%, 5% |
| reps per cell | 80, one new phenotype per rep on a fixed per-cell genotype design |

That is 72 main cells. A further 16 cells make the contamination experiment
(see below). The grid has 88 cells x 80 reps = 7040 tasks.

SNP sets. K comes from 1500 SNPs. The test set is a disjoint set of 2000
SNPs, so the Bonferroni threshold is 0.05/2000 = 2.5e-5. Both sets contain
only complete-call SNPs with MAF >= 0.05, because pylmm's `GWAS` re-fits per
missing-genotype pattern and that would confound the refit comparison. In the
BED data about half of the SNPs have a missing call, so they are dropped.
Causal SNPs are drawn at random inside the test set each replicate, so they
are never in K in the main analysis. The other 1999 or 1995 test SNPs are
the non-causal ("null") SNPs.

Methods (all see the same y):

* `lmm`: pylmm null fit (REML) once, then `L.association` for all test SNPs.
  This is what `lmm.GWAS(refit=False)` does (the `REML` flag never reaches the
  null fit: bug 1; here the null fit is REML and the association REML).
* `lmmr`: `L.fit(X=x)` then `L.association(x)` for every SNP (what
  `GWAS(refit=True)` does), but only on the causal SNPs plus a random sample
  of 100 non-causal SNPs per rep. A refit costs about 4.5 ms per SNP.
* `ols`: per-SNP OLS t-test (`methods.ols_gwas`) and slope, intercept only.
* `orc`: independent dense oracle Wald with REML h on the causal SNPs, for
  unrelated and balding_nichols at h2 = 0.5 (n = 500). It is a reference and a
  correctness check: its beta and t should equal pylmm's.
  Quick run: beta-hat is identical to pylmm's, as expected.

The `beta`-hat comes from `LMM.association(..., returnBeta=True)`.

Contamination experiment: balding_nichols and BED, h2 = 0.5, 5 causal SNPs,
4 effect sizes, with K from only 500 SNPs (a small K makes any contamination
visible). `contam=0` builds K from the 500 SNPs. `contam=1` rebuilds K every
rep from those 500 SNPs plus the 5 causal SNPs, as happens when K is built
from all genotyped SNPs. Only the proximal-contamination effect on power and
on beta-hat bias is of interest here; both arms run pylmm fixed h, refit and
OLS (OLS is unaffected by K and is a sanity control).

## Metrics

For each cell and method, pooled over reps:

* power at alpha = 5e-8, 1e-5 and 0.05/m_test, with Wilson 95% CIs (k causal
  hits / total causal SNPs);
* empirical type-I error on the non-causal SNPs at the same three thresholds
  (counts from all 2000 non-causal SNPs for `lmm` and `ols`, from the
  100-SNP sample for `lmmr`; the 5e-8 rate is usually 0 and is reported with
  its CI);
* lambda_GC on the non-causal SNPs (`lmm`, `ols`);
* power at an empirically calibrated threshold: the 1e-3 (also 0.01 in the CSV)
  quantile of the pooled non-causal p-values of that method and cell, i.e. the
  power each method would have if it were calibrated to type-I error 1e-3.
  Using the pooled null sample (100 per rep x 80 reps = 8000 per cell) gives
  about 8 null SNPs at 1e-3, so the calibrated power is noisy but fair across
  methods. Calibrating at finer thresholds like 5e-8 is not possible from this
  sample;
* effect-size accuracy: relative bias `mean(beta_hat/beta - 1)`, relative RMSE
  `sqrt(mean((beta_hat/beta - 1)^2))`, and the relative bias among causal SNPs
  with p <= 1e-5 (the winner's-curse bias; empty cells mean no SNP was
  significant).
  The paired difference `dlogp_refit` and `dbeta_refit` (refit minus fixed h,
  same SNPs) is in the raw rows.

Caveat on CIs: the causal SNPs of one rep share a phenotype, and all reps of
a cell share genotypes and K, so the Wilson CIs are somewhat optimistic.
With 1 causal SNP per rep a cell has only 80 observations (CI about +-0.10
at power 0.5).

Outputs (`just sim-report s3_power`): `summary.csv`/`summary.md` (compact
table plus the contamination table), `power_table.csv` (everything, with CIs),
`power_<source>.png` (power vs effect size: rows = raw power at Bonferroni and
calibrated power at 1e-3, columns = h2, colour = method, solid = 5 causal,
dashed = 1 causal) and `type1_1e-05.png`.

## Hypotheses

* H1. On unrelated genotypes, pylmm and OLS have the same power (K carries no
  information), at all h2, and both are calibrated.
* H2. With structure (balding_nichols, BED), OLS has high raw power but also
  type-I error far above nominal (lambda_GC > 1). At a calibrated threshold OLS
  has equal or less power than pylmm, increasingly so as h2 rises. Raw OLS
  power is therefore not a gain.
* H3. pylmm's null rejection stays close to nominal for balding_nichols. For BED
  (rank-deficient, replicate inbred animals, strong LD) the quick run showed
  lambda_GC well above 1 even for pylmm, so its raw power also contains false
  positives there; the calibrated column separates the two.
* H4. Power rises with `vpe`. For a fixed `vpe` it should change little with
  h2 for pylmm (the LMM's residual variance is about (1 - vpe_total) either
  way) but fall for OLS under structure, where u acts as extra noise. Five
  causal SNPs at the same `vpe` should give about the same per-SNP power as
  one.
* H5. Refit vs fixed h: little difference in power or beta-hat when the SNP
  is small (the null h is already close to the per-SNP h), and a small
  advantage or disadvantage for large `vpe` where the causal SNP moves the REML
  h estimate. Refit h is estimated from the full model, so the p-value
  should be slightly better calibrated for strongly associated SNPs.
* H6. Proximal contamination (`contam=1`) lowers power and shrinks beta-hat
  (the SNP's own effect is partly absorbed into u), more for larger `vpe`
  and smaller K SNP sets.
* H7. Beta-hat is unbiased on average in all methods, but conditional on
  significance it is upward-biased, strongest at small `vpe`.

## Interpretation guide (to fill in after the full run)

* Compare `pow_cal1e-3` between OLS and pylmm within a (source, h2) block, not
  `pow_bonf`. If `t1e_1e-5` for OLS is orders of magnitude above 1e-5, its raw
  power is not evidence.
* For refit vs fixed, look at the overlap of the Wilson CIs in
  `power_table.csv`, and at `rel_bias`/`rel_rmse`. Remember the refit rows use
  the same causal SNPs and phenotypes as the fixed-h rows, so differences are
  paired even though the CIs are not.
* Contamination: compare `power_bonf_contam1` vs `power_bonf_contam0` in the
  contamination table; the OLS rows in `power_table.csv` should not move.
* If pylmm's null rejection on BED is itself inflated, report power only at
  the calibrated threshold for that source.

## Compute

Measured on one core (full-size cells, n = 500/600, 2000 test SNPs): about
0.7 s per replicate for lmm + refit + OLS, 1.3 s with the oracle, 1.0 s for the
contamination cells (per-rep K and eigh). BED adds a 1.5 s genotype load per
worker and cell group (cached). Total about 80 x (16 x 1.3 + 56 x 0.75 +
16 x 1.05) = 6400 core-seconds, so about 18 min at `--jobs 6` if scaling were
perfect, and an estimated 20-25 min with the usual contention. Memory per
worker is small (< 300 MB). The `--quick` grid (4 cells x 3 reps, n = 300) runs
in about 6 s wall with 2 jobs.

## Known limitations

* Only complete-call SNPs with MAF >= 0.05 are used, which differs from a real
  BED GWAS with missing calls.
* The test SNPs are independent within the synthetic sources; BED test SNPs
  have real LD (which does not matter for per-SNP power, but means null SNPs
  are correlated, so null rates have larger sampling error than the binomial CI
  suggests).
* Only equal effects across causal SNPs, one MAF filter, and n fixed per
  source (500/500/600), so sample size is not varied.
* Raw 5e-8 type-I error cannot be measured with 2000 nulls x 80 reps for
  pylmm if it is well calibrated; it is reported as a count with an
  upper bound only.
