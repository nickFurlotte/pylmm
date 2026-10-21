# S7: the h search on [0, 1] (Phase 6.3) vs pylmm 1.0.0's

Study code: `sims/studies/s7_numerics.py`. Full tables: `sims/results/s7_numerics/summary.md` /
`summary.csv`. Reproduce: `just sim s7_numerics --jobs 6 --report` (960 replicates, about 2 min at 6
jobs on an 8-core laptop); `just sim s7_numerics --quick` runs in seconds. The real-data table at the
end: `uv run --group sims python -m sims.studies.s7_numerics 20000`.

**Question.** Does the default search (grid 0, 0.01, ..., 1; every local maximum refined, the ends
included; the best kept; the likelihood evaluated on an orthonormal basis of the design) find the
maximum of the likelihood where 1.0.0's (`legacy_fit=True`: grid to 0.99, the first interior peak below
0.98, else the higher end unrefined) did not, and does it leave the null calibration of the scan as it
was?

**Design.** 4 cohorts (`unrelated300`: a centred GRM of rank n - 1 from 1000 SNPs; `family300`: 100
families of 3 sibs; `bed300` / `bed60`: random mice of the bundled inbred panel, K from 2500 SNPs, rank
about 100) x true h² in {0, 0.05, 0.5, 0.95, 0.98, 1} x 40 replicates, y = u + e drawn exactly from K.
Each replicate: REML and ML null fits, default and legacy, on the same eigendecomposition; the oracle
(`sims/oracle.py`: dense Cholesky, its own 201-point global search, with pylmm's 1e-6 eigenvalue floor so
both maximise the same function); a REML scan of 1000 null test SNPs with both fits. (Run on the code
after Phase 6.6, whose kinship and phenotype generators changed the simulated data.)

## Results (full run)

* **The default search reaches the oracle's maximum in every fit** (960 x REML and ML): the largest
  log-likelihood shortfall is 4.5e-12 (REML) and 3.9e-12 (ML). Where |h - h_oracle| reaches 2.3e-3 (a
  few cells) the profile is flat there: the likelihoods agree to 1e-6.
* **1.0.0 missed it near 1**: REML, share of fits more than 1e-6 below the maximum (largest shortfall):

  | true h² | unrelated300 | family300 | bed300 | bed60 |
  |---|---|---|---|---|
  | 0 | 0 | 0.05 (0.001) | 0.03 (0.02) | 0 |
  | 0.95 | 0.47 (0.10) | 0.28 (0.23) | 0 | 0 |
  | 0.98 | 0.55 (0.21) | 0.53 (0.33) | 0.85 (11) | 0.62 (4.1) |
  | 1 | 0.62 (0.19) | 0.78 (0.35) | 1.00 (*) | 1.00 (*) |

  (*) At true h² = 1 on the inbred panel the default returns h = 1, as the oracle does; 1.0.0 stopped at
  0.99. The likelihood gap there (950 and 74) is not a property of the data: with a singular K the
  exact likelihood rises without bound as h -> 1 along K's null space, and the value at h = 1 is set by
  pylmm's 1e-6 eigenvalue floor (about half the null dimension times log 1e6), which the oracle copies.
  The h² <= 0.98 gaps are the data's. At h² = 0 the misses are 1.0.0's unrefined end h = 0 with the
  maximum inside [0, 0.01].
* **ML**: on the outbred cohorts (centred GRM, m > n) the ML likelihood has no maximum (bug 17) and
  every default ML fit warns; there "the maximum" is the floor's, and both searches are measured against
  it only to check the search. On the inbred panel (bounded ML) 1.0.0 missed the ML maximum in 85% (bed300)
  and 65% (bed60) of the fits at h² = 0.98 and all at h² = 1.
* **ML sigma** is now Q/n, the oracle's (relative error <= 9e-6); 1.0.0 reported Q/(n - p).
* **Null calibration** (REML scan, mean λ_GC and type-I error at 0.05 / 1e-3) is unchanged wherever 1.0.0
  found the maximum, and better where it did not: bed300 at h² = 0.98, λ_GC 0.705 -> 1.054 and type-I
  error at 0.05 0.019 -> 0.053; bed60 at 0.98, 0.930 -> 1.008. max |Δ−log10 p| per replicate is 0 to 1.4
  (largest where 1.0.0 stopped at the 0.99 cap).
* **h² = 1 on the inbred panel** inflates both searches about equally (bed300: λ_GC 3.57 default, 3.50
  legacy; bed60: 1.37 / 1.36). With no residual variance y lies in the span of K (rank about 100), and
  the Wald test's n - p degrees of freedom overstate the information: a model boundary, not the search.
  The default fit now warns there ("h is at its upper boundary").
* **Fit time**: median null fit 3.8-8.9 ms (default) vs 2.3-5.7 ms (legacy), n = 60-300, 6 parallel
  jobs: the default refines more brackets and evaluates 101 grid points instead of 100. In a single
  process (n = 300 and 1000) one null fit takes 3-4 ms either way.

## Real data: multi-modal profiles

The REML profiles in the simulations had one maximum (plus, for ML on the outbred cohorts, the spike at
h = 1). On real data, of the 3772 distinct subsets of HMDP strains that `gwas` fits for SNPs with missing
calls (the first 20,000 HMDP SNPs), one has two maxima: SNP 110 (58 strains), an interior peak at h = 0.223
(LL 7.601) and a higher boundary maximum at h = 0 (LL 7.630). 1.0.0 kept the first; the default keeps the
best, as the oracle does. That SNP's t moves from 0.762 to 0.824
(`tests/test_association.py::test_gwas_hmdp_default_differs_only_where_legacy_kept_a_worse_optimum`).

The README example data (1219 mice, about 100 strains) has its REML maximum at h = 0.0038, inside
[0, 0.01], where 1.0.0 returned the unrefined end h = 0; see the CHANGELOG for how the scan moves.
