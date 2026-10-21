# S8: leave-one-chromosome-out (LOCO) vs the all-SNP kinship

Question: does a leave-one-chromosome-out scan (`pylmm.loco.gwas_loco`, `pylmm gwas --loco`) fix the
conservative null calibration of a kinship built from all SNPs (S6: λ_GC ~0.7 on inbred panels once
h2 > 0), and what does it do to power? On outbred and inbred genotype sources, with a polygenic
background spread over many chromosomes.

Code: `sims/studies/s8_loco.py`. Run with `just sim s8_loco --jobs 6 --report` (`--quick`: two small
cells, about a second). Results: `sims/results/s8_loco/summary.{md,csv}`. Timing and the README data:
`just bench loco` (`sims/bench/loco.py`), below.

## Design

| factor | levels |
|---|---|
| panel | `family`: 300 outbred individuals in 100 sib trios (`sims.genotypes.family`), 8000 unlinked SNPs labelled with 10 chromosomes of 800, no missing calls; `inbred_panel`: 200 synthetic RI-like strains (8 founders, 10 chromosomes of 1000 linked SNPs, 2% missing calls); `hmdp`: the bundled 94 HMDP strains, every 6th of the 132,285 SNPs (22,048; 2.1% missing), chromosomes 1-19 and X from the marker map |
| h2 | 0, 0.3, 0.6, 0.9: the share of the polygenic background |
| background | 300 causal SNPs with per-SD effects, drawn per replicate from the **odd** autosomes (MAF >= 0.05, mean-imputed) |
| QTL | one more SNP on an odd autosome, explaining q2 = 0.05; noise 1 - h2 - q2 |
| tested SNPs | up to 1500 null SNPs on the **even** autosomes (no causal SNP on their chromosome), the QTL, and 100 of the background's causal SNPs |
| kinships | `all`: the standardized kinship of all SNPs (missing calls included), one eigendecomposition, `gwas`; `loco`: `LocoKinship` of the same SNPs, `gwas_loco` (a null fit and an eigendecomposition per chromosome) |
| replicates | 100 per cell (12 cells, 1200 replicates; 147 s on 6 workers, on a loaded machine) |

Per replicate and method: REML null h (`loco`: the mean over the tested chromosomes), λ_GC and type-I
error over the null SNPs, −log10 p of the QTL and its power at 1e-3 and 1e-5, and the mean χ² (t²) of the
tested background causal SNPs. pylmm's default fit (the h search of Phase 6.3) and Wald test throughout (null fit once
per K). Before #27 merged the same study gave the same numbers to within 0.003 in λ_GC.

## Results (full run, `sims/results/s8_loco/summary.md`)

Null calibration over the null SNPs (mean over 100 replicates; type-I at 0.05 / 1e-3):

| panel | h2 | λ_GC all | λ_GC LOCO | t1e 0.05 all / LOCO | t1e 1e-3 all / LOCO |
|---|---|---|---|---|---|
| family | 0 | 0.984 | 0.987 | 0.048 / 0.048 | 0.0008 / 0.0008 |
| family | 0.3 | 0.997 | 1.011 | 0.049 / 0.051 | 0.0009 / 0.0010 |
| family | 0.6 | 0.973 | 0.989 | 0.048 / 0.050 | 0.0010 / 0.0011 |
| family | 0.9 | 0.967 | 1.001 | 0.046 / 0.051 | 0.0008 / 0.0010 |
| inbred_panel | 0 | 0.966 | 1.032 | 0.045 / 0.052 | 0.0006 / 0.0009 |
| inbred_panel | 0.3 | **0.791** | 1.022 | 0.027 / 0.050 | 0.0001 / 0.0011 |
| inbred_panel | 0.6 | **0.660** | 1.036 | 0.015 / 0.049 | 0.00005 / 0.0009 |
| inbred_panel | 0.9 | **0.564** | 1.033 | 0.009 / 0.050 | 0.00002 / 0.0008 |
| hmdp | 0 | 0.953 | 0.976 | 0.043 / 0.046 | 0.0009 / 0.0010 |
| hmdp | 0.3 | 0.940 | 1.017 | 0.040 / 0.048 | 0.0005 / 0.0008 |
| hmdp | 0.6 | **0.853** | 0.990 | 0.034 / 0.048 | 0.0005 / 0.0013 |
| hmdp | 0.9 | **0.763** | 1.014 | 0.025 / 0.048 | 0.0003 / 0.0010 |

Power (the QTL, q2 = 0.05; mean −log10 p, share of replicates with p ≤ 1e-3) and the mean χ² of the
background's causal SNPs:

| panel | h2 | QTL −log10 p all / LOCO | power 1e-3 all / LOCO | causal χ² all / LOCO | null h all / LOCO |
|---|---|---|---|---|---|
| family | 0.3 | 3.67 / 3.71 | 0.61 / 0.62 | 1.27 / 1.29 | 0.365 / 0.362 |
| family | 0.9 | 4.07 / 4.18 | 0.74 / 0.75 | 1.84 / 1.91 | 0.932 / 0.925 |
| inbred_panel | 0.3 | 1.98 / 2.46 | 0.17 / 0.36 | 1.37 / 1.94 | 0.336 / 0.303 |
| inbred_panel | 0.6 | 2.47 / 3.82 | 0.30 / 0.55 | 1.63 / 3.16 | 0.648 / 0.583 |
| inbred_panel | 0.9 | 2.64 / 4.54 | 0.32 / 0.58 | 1.97 / 5.07 | 0.937 / 0.847 |
| hmdp | 0.6 | 1.37 / 1.63 | 0.04 / 0.12 | 1.32 / 1.67 | 0.629 / 0.613 |
| hmdp | 0.9 | 1.76 / 2.30 | 0.20 / 0.35 | 1.54 / 2.55 | 0.943 / 0.930 |

Findings:

1. **LOCO is calibrated everywhere** (λ_GC 0.976-1.036, type-I at 0.05 0.046-0.052, at 1e-3
   0.0008-0.0013), for every panel and h2. The all-SNP K is conservative, increasingly with h2, and
   strongly so in inbred panels (λ 0.56 on the synthetic panel at h2 = 0.9; type-I at 1e-3 down to 2e-5,
   50x below nominal), as S6 found with a coarser LOCO (one K without all tested chromosomes).
2. **LOCO gains power, a lot in inbred panels.** At the same q2, the QTL's power at 1e-3 doubles on the
   synthetic panel (0.17 → 0.36, 0.30 → 0.55, 0.32 → 0.58) and on the HMDP strains at h2 0.6-0.9; the
   background's causal SNPs' mean χ² rises 1.4-2.6x there. On outbred families the gain is small
   (mean −log10 p +0.04 to +0.11) because their SNPs are unlinked: only the tested SNP itself is in K.
3. **The null h is a little lower with LOCO** when the trait is polygenic (synthetic panel, h2 = 0.9:
   0.85 vs 0.94): `K_{-c}` misses the background on chromosome c. Each chromosome gets its own h.
4. At h2 = 0 the two agree to within noise (λ 0.95-1.03); LOCO costs nothing there.
5. Outbred and unlinked (family): the all-SNP deflation is mild (λ 0.97 at h2 = 0.9) and LOCO removes it.
   With LD, more of each chromosome is in K along with the tested SNP, so outbred data with real LD
   should fall between the family and inbred rows (not simulated here).

## The README data (`just bench loco`)

`pylmm kinship` + `pylmm gwas --kinship` vs `pylmm gwas --loco` on the bundled 1219 mice x 98,740 SNPs,
19 autosomes (macOS arm64, 1 BLAS thread, wall time including start-up and I/O; two runs, the second on a
machine busy with other jobs):

| step | run 1 (s) | run 2 (s) |
|---|---|---|
| standard: `pylmm kinship` | 5.8 | 8.3 |
| standard: `pylmm gwas --kinship` | 8.1 | 12.3 |
| standard: total | 13.9 | 20.6 |
| `pylmm gwas --loco` | 16.5 | 20.1 |
| (of which: building the 19 LOCO kinships, one pass) | 4.2 | 5.3 |

So LOCO costs about what the two-step standard run costs (1.0-1.2x here): building every `K_{-c}` costs
what one `pylmm kinship` costs, and the 18 extra eigendecompositions and null fits (about 0.2 s each at
n = 1219) are the rest. The eigendecompositions grow as n^3 x chromosomes, the kinship pass as n^2 x m.

The README phenotype is fake (null h 0.004 with all SNPs, 0.002-0.005 with LOCO): λ_GC 1.050 standard,
1.079 LOCO; SNPs with p < 1e-3: 18 vs 24, none below 1e-5; the top SNP is the same (mm37-7-19743810,
p 2.1e-4 both), and the top 10 share 7 SNPs (max |Δ−log10 p| 0.11, median 0.004). A seeded polygenic
trait simulated on the same genotypes (h2 = 0.5 from 300 SNPs on the odd autosomes) shows the inbred
deflation on real mouse genotypes: over the null even autosomes λ_GC 0.769 with K from all SNPs and
0.946 with LOCO (type-I at 0.05: 0.019 vs 0.040).

## Not covered

* The Bennett / Farber HMDP data of the replication (`docs/replication.md`) are not in the repository;
  the replication's hand-built LOCO there moved −log10 p by −0.6 to +1.2.
* Dosages, n in the thousands (memory is one n × n matrix per chromosome), and LOCO with `--refit`
  (supported; not simulated).
