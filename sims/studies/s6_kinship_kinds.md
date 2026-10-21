# S6: standardized, centred and IBS kinships on inbred-strain panels

Question: on an inbred panel (strains related through a few founders, 0/1 genotypes, missing calls), how
do the three kinships of `pylmm.kinship` (`kind="standardized"`, `"centered"`, `"ibs"`) compare for
heritability recovery and null calibration? Does the kinship scale (`kinship_scale`) put their `h` on a
common PVE scale?

Code: `sims/studies/s6_kinship_kinds.py`. Run with `just sim s6_kinship_kinds --jobs 6`, then
`just sim-report s6_kinship_kinds` (`--quick`: one small cell pair, about a second). Results:
`sims/results/s6_kinship_kinds/summary.{md,csv}`.

## Design

| factor | levels |
|---|---|
| panel | `inbred_panel`: 200 synthetic RI-like strains (`sims.genotypes.inbred_panel`: mosaics of 8 founders, 10 chromosomes of 1200 SNPs, Poisson(2) breakpoints per chromosome, 2% missing calls, seed 1); `hmdp`: the bundled 94 HMDP strains, every 6th of the 132,285 SNPs (22,048; 2.1% missing), chromosomes from the marker map |
| h2 | 0, 0.3, 0.6 (realised genetic share of the sample variance) |
| effect | `standardized` (effects per SD of the genotype: the GRM's model) or `per_allele` (per allele: the centred / IBS model) |
| causal SNPs | 200, drawn per replicate from the first half of the autosomes (MAF >= 0.05, mean-imputed) |
| tested SNPs | every SNP with MAF >= 0.05 on the second half of the autosomes (null: unlinked to every causal SNP) |
| replicates | 100 per cell (12 cells) |

The kinships are built from all SNPs (missing calls in each kind's own convention), so none is the
generating covariance: as with real data, each is an approximation. `std_loco` / `ibs_loco` are the
standardized and IBS kinships built without the tested chromosomes (what a LOCO scan uses). Per
replicate and kinship: REML `h`; `pve = h c / (h c + 1 - h)` with `c` the centred mean diagonal; λ_GC and
type-I error over the null SNPs (pylmm's default null-fit Wald test).

## Results (full run, 1200 replicates, 156 s on 6 workers)

Heritability (means over 100 replicates; SE of a mean about 0.005-0.01):

| panel | h2 | effect | pve_true | h std | pve std | h cen | pve cen | h ibs | pve ibs | pve std_loco | pve ibs_loco |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inbred_panel | 0 | standardized | 0 | 0.047 | 0.046 | 0.158 | 0.044 | 0.112 | 0.045 | 0.030 | 0.030 |
| inbred_panel | 0 | per_allele | 0 | 0.037 | 0.037 | 0.129 | 0.037 | 0.092 | 0.038 | 0.025 | 0.026 |
| inbred_panel | 0.3 | standardized | 0.302 | 0.299 | 0.296 | 0.670 | 0.293 | 0.555 | 0.299 | 0.300 | 0.305 |
| inbred_panel | 0.3 | per_allele | 0.300 | 0.290 | 0.287 | 0.661 | 0.285 | 0.546 | 0.291 | 0.293 | 0.297 |
| inbred_panel | 0.6 | standardized | 0.611 | 0.588 | 0.584 | 0.877 | 0.582 | 0.814 | 0.593 | 0.590 | 0.600 |
| inbred_panel | 0.6 | per_allele | 0.604 | 0.609 | 0.606 | 0.888 | 0.605 | 0.830 | 0.617 | 0.607 | 0.619 |
| hmdp | 0 | standardized | 0 | 0.047 | 0.047 | 0.164 | 0.054 | 0.117 | 0.055 | 0.044 | 0.046 |
| hmdp | 0 | per_allele | 0 | 0.041 | 0.040 | 0.137 | 0.043 | 0.097 | 0.044 | 0.033 | 0.036 |
| hmdp | 0.3 | standardized | 0.305 | 0.268 | 0.266 | 0.603 | 0.279 | 0.490 | 0.285 | 0.259 | 0.268 |
| hmdp | 0.3 | per_allele | 0.309 | 0.260 | 0.258 | 0.576 | 0.273 | 0.471 | 0.278 | 0.264 | 0.276 |
| hmdp | 0.6 | standardized | 0.612 | 0.596 | 0.593 | 0.882 | 0.608 | 0.813 | 0.620 | 0.586 | 0.604 |
| hmdp | 0.6 | per_allele | 0.603 | 0.588 | 0.585 | 0.876 | 0.614 | 0.809 | 0.623 | 0.577 | 0.605 |

Null calibration (λ_GC, type-I error at 0.05 and 0.01):

| panel | h2 | effect | λ std | λ cen | λ ibs | λ std_loco | λ ibs_loco | t1e .05 std / ibs / ibs_loco | t1e .01 std / ibs / ibs_loco |
|---|---|---|---|---|---|---|---|---|---|
| inbred_panel | 0 | standardized | 0.986 | 0.987 | 0.987 | 1.036 | 1.036 | 0.046 / 0.046 / 0.051 | 0.008 / 0.008 / 0.010 |
| inbred_panel | 0 | per_allele | 0.966 | 0.966 | 0.966 | 1.005 | 1.005 | 0.045 / 0.045 / 0.049 | 0.008 / 0.008 / 0.009 |
| inbred_panel | 0.3 | standardized | 0.803 | 0.804 | 0.805 | 0.992 | 0.993 | 0.028 / 0.029 / 0.048 | 0.004 / 0.004 / 0.010 |
| inbred_panel | 0.3 | per_allele | 0.802 | 0.803 | 0.803 | 1.000 | 1.001 | 0.027 / 0.027 / 0.048 | 0.004 / 0.004 / 0.009 |
| inbred_panel | 0.6 | standardized | 0.684 | 0.686 | 0.686 | 1.009 | 1.009 | 0.017 / 0.018 / 0.048 | 0.002 / 0.002 / 0.009 |
| inbred_panel | 0.6 | per_allele | 0.677 | 0.678 | 0.678 | 1.014 | 1.015 | 0.018 / 0.018 / 0.050 | 0.002 / 0.002 / 0.010 |
| hmdp | 0 | standardized | 0.916 | 0.913 | 0.913 | 0.932 | 0.933 | 0.040 / 0.040 / 0.042 | 0.008 / 0.007 / 0.008 |
| hmdp | 0 | per_allele | 0.926 | 0.922 | 0.922 | 0.946 | 0.946 | 0.039 / 0.038 / 0.041 | 0.007 / 0.007 / 0.008 |
| hmdp | 0.3 | standardized | 0.967 | 0.958 | 0.958 | 1.058 | 1.060 | 0.045 / 0.044 / 0.054 | 0.008 / 0.008 / 0.011 |
| hmdp | 0.3 | per_allele | 0.934 | 0.924 | 0.924 | 1.000 | 1.003 | 0.041 / 0.040 / 0.049 | 0.007 / 0.007 / 0.009 |
| hmdp | 0.6 | standardized | 0.880 | 0.881 | 0.879 | 1.038 | 1.043 | 0.036 / 0.037 / 0.050 | 0.006 / 0.006 / 0.010 |
| hmdp | 0.6 | per_allele | 0.874 | 0.876 | 0.875 | 1.028 | 1.031 | 0.037 / 0.036 / 0.053 | 0.006 / 0.006 / 0.011 |

## Findings

1. **`h` is on each kinship's scale; PVE is comparable.** For the same data the fitted `h` differs by
   kind (inbred panel, h2 = 0.3: 0.30 standardized, 0.67 centred, 0.55 IBS), because the centred mean
   diagonal `c` differs (about 0.98, 0.18 and 0.32 on the synthetic panel). The PVE `h c / (h c + 1 - h)`
   agrees across the three kinds within 0.01-0.04 and with the true genetic share (bias -0.05 to +0.02; the larger negative ones on the 94 HMDP strains).
   Report PVE, or `h` with its kinship, never a bare `h` from an IBS kinship next to a GRM `h`.
2. **The kind barely matters for the tests.** λ_GC and type-I error of the three kinds agree to 0.01 in
   every cell, whichever effect model generated the phenotype; the IBS kinship is not better calibrated
   than the standardized one on these panels, nor worse. (Power was not measured here; the replication's
   finding that IBS moved p-values toward EMMA-era papers by +0.07 to +0.41 −log10 p is about matching
   their method, not about calibration.)
3. **K from all SNPs is conservative on inbred panels when h2 > 0, for every kind** (λ 0.68-0.80 on the
   synthetic panel, 0.87-0.97 on HMDP; t1e at 0.05 down to 0.017): the tested chromosomes in K absorb
   variance the phenotype does not have along them. Without the tested chromosomes in K (`*_loco`), λ is
   0.99-1.06 and t1e 0.048-0.054. LOCO, not the kinship kind, is the calibration lever (the CLI has no
   LOCO flag yet; see `custom-association-analysis`).
   Context: in this design the LOCO kinship is built from exactly the chromosomes that carry the causal
   SNPs, so it is close to the generating covariance and λ ≈ 1 is expected by construction; the
   all-SNP deflation is the finding. It is not an artefact of the half/half split: the PR #26 review
   re-ran inbred_panel, h2 = 0.6 (30 replicates) with the causal SNPs on chromosomes 1-9 and the tests on
   chromosome 10, and got λ 0.703 with K from all SNPs and 1.075 with LOCO (half/half: 0.691 / 1.018).
   It is proximal contamination, strong in an 8-founder RI-like panel, where each chromosome's SNPs are
   a large share of the strain relatedness in K.
4. **h2 = 0 bias**: REML `h` at h2 = 0 is 0.04-0.05 (standardized), larger on the other kinds' scales, a
   boundary effect (h ≥ 0); PVE 0.03-0.06 for all.
5. **HMDP at h2 = 0 is mildly conservative for every kinship, LOCO included** (λ 0.91-0.95, t1e at 0.05
   0.038-0.042; synthetic panel at h2 = 0: λ 0.97-1.04). Not investigated further; a likely reading is
   that with 94 strains the null SNPs lie largely along the leading eigenvectors of K, which the fitted
   h (0.03-0.16 on the kinships' scales, from the h ≥ 0 boundary) down-weights, and that the median of
   7,416 strongly linked SNPs per replicate is noisy. S1 also found the test mildly conservative
   at h2 = 0. The kind makes no difference here either.

Interpretation for users of EMMA-era inbred-panel studies: `kind="ibs"` reproduces their kinship; it does
not change calibration, and its `h` must be converted to PVE (or to the other kinship's scale) before it
is compared with a GRM-based heritability.
