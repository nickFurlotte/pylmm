# S8: leave-one-chromosome-out (LOCO) vs the all-SNP kinship

Question: what does a leave-one-chromosome-out scan (`pylmm.loco.gwas_loco`, `pylmm gwas --loco`) do
to calibration and power, compared with one kinship built from all SNPs? S6 found that the all-SNP
kinship is conservative on inbred panels once h2 > 0 (λ_GC ~0.7). On outbred and inbred genotype
sources, with a polygenic background either away from the tested chromosomes or, as in real traits,
on every chromosome.

Code: `sims/studies/s8_loco.py`. Run with `just sim s8_loco --jobs 6 --report` (`--quick`: four small
cells, about a second). Results: `sims/results/s8_loco/summary.{md,csv}`. Timing and the README data:
`just bench loco` (`sims/bench/loco.py`), below.

## Design

| factor | levels |
|---|---|
| design | `split`: background and QTL on the **odd** autosomes, tested SNPs on the **even** ones (no causal SNP on their chromosome); `all`: background and QTL on **every** autosome, tested SNPs are non-causal SNPs from every autosome |
| panel | `family`: 300 outbred individuals in 100 sib trios (`sims.genotypes.family`), 8000 **unlinked** SNPs labelled with 10 chromosomes of 800, no missing calls; `inbred_panel`: 200 synthetic RI-like strains (8 founders, 10 chromosomes of 1000 linked SNPs, 2% missing calls); `hmdp`: the bundled 94 HMDP strains, every 6th of the 132,285 SNPs (22,048; 2.1% missing), chromosomes 1-19 and X from the marker map |
| h2 | 0, 0.3, 0.6, 0.9: the share of the polygenic background |
| background | 300 causal SNPs with per-SD effects, drawn per replicate (MAF >= 0.05, mean-imputed) |
| QTL | one more causal SNP, explaining q2 = 0.05; noise 1 - h2 - q2 |
| tested | up to 1500 non-causal SNPs (as the design says), the QTL, 100 of the background's causal SNPs |
| iid null (`all` only) | a separate N(0, 1) phenotype scanned at the same SNPs with the same kinships: the calibration check |
| kinships | `all`: the standardized kinship of all SNPs (missing calls included), one eigendecomposition, `gwas`; `loco`: `LocoKinship` of the same SNPs, `gwas_loco` (an eigendecomposition and a null fit per chromosome) |
| replicates | 100 per cell (24 cells, 2400 replicates; 218 s on 6 workers) |

`split` is where LOCO is right by construction: `K_{-c}` of an even chromosome holds the whole
background, so it is the generating covariance. (The PR #26 review raised the same caveat about S6.)
`all` is the realistic case. There the non-causal SNPs on chromosome c tag the causal SNPs of c through
LD, and `K_{-c}` no longer explains them, so under LOCO they are associated: their λ_GC is not a
calibration measure. The iid null phenotype is.

Per replicate and method: REML null h (`loco`: the mean over chromosomes), λ_GC and type-I error over
the tested non-causal SNPs, −log10 p of the QTL and its power at 1e-3, and the mean χ² (t²) of the
tested background causal SNPs. For `all`, also λ_GC and type-I error of the iid null phenotype. pylmm's
default fit (the Phase 6.3 h search) and Wald test throughout.

## Results (full run, `sims/results/s8_loco/summary.md`; mean over 100 replicates)

### Calibration: the iid null phenotype (design `all`)

| panel | h2 of the *other* trait | λ_GC all / LOCO | type-I 0.05 all / LOCO | type-I 1e-3 all / LOCO |
|---|---|---|---|---|
| family | (any) | 0.971-0.983 / 0.973-0.985 | 0.046-0.048 / 0.047-0.048 | 0.0008-0.0010 / 0.0008-0.0010 |
| inbred_panel | (any) | 0.964-0.969 / 1.004-1.018 | 0.045-0.046 / 0.049-0.052 | 0.0008 / 0.0009-0.0012 |
| hmdp | (any) | 0.924-0.941 / 0.941-0.956 | 0.039-0.040 / 0.040-0.042 | 0.0004-0.0006 / 0.0005-0.0007 |

(The iid phenotype is independent of the trait, so the four h2 cells are four repeats; ranges over them.)
On a phenotype with no genetic signal, both scans are calibrated. LOCO is slightly closer to nominal.
Both are mildly conservative on the 94 HMDP strains, as in S6.

### λ_GC over the tested non-causal SNPs

| panel | h2 | `split` (other chromosomes) all / LOCO | `all` (every chromosome) all / LOCO |
|---|---|---|---|
| family | 0.3 | 0.987 / 0.997 | 0.990 / 1.003 |
| family | 0.9 | 0.968 / 0.997 | 0.964 / 1.001 |
| inbred_panel | 0.3 | **0.782** / 1.017 | 0.990 / **1.345** |
| inbred_panel | 0.6 | **0.662** / 1.030 | 0.975 / **1.850** |
| inbred_panel | 0.9 | **0.571** / 1.029 | 0.951 / **2.805** |
| hmdp | 0.3 | 0.917 / 1.006 | 1.014 / 1.129 |
| hmdp | 0.6 | 0.884 / 1.056 | 1.002 / 1.239 |
| hmdp | 0.9 | **0.788** / 1.047 | 0.992 / **1.525** |

Type-I error in `split`, at 0.05 / 1e-3: all-SNP K down to 0.010 / 0.000013 (inbred, h2 = 0.9), LOCO
0.044-0.053 / 0.0006-0.0014 in every cell.

### Power: the QTL (q2 = 0.05) and the background's causal SNPs

| design | panel | h2 | QTL power at 1e-3 all / LOCO | QTL mean −log10 p | causal SNPs' mean χ² | null h all / LOCO |
|---|---|---|---|---|---|---|
| split | inbred_panel | 0.6 | 0.30 / 0.48 | 2.44 / 3.58 | 1.67 / 3.17 | 0.649 / 0.584 |
| split | inbred_panel | 0.9 | 0.42 / 0.68 | 2.90 / 5.01 | 1.94 / 5.02 | 0.941 / 0.845 |
| split | hmdp | 0.9 | 0.10 / 0.21 | 1.52 / 1.97 | 1.50 / 2.42 | 0.941 / 0.928 |
| all | family | 0.9 | 0.70 / 0.73 | 4.31 / 4.44 | 1.84 / 1.91 | 0.916 / 0.909 |
| all | inbred_panel | 0.3 | 0.29 / 0.43 | 2.38 / 2.96 | 1.17 / 1.57 | 0.324 / 0.291 |
| all | inbred_panel | 0.6 | 0.37 / 0.59 | 2.48 / 3.71 | 1.32 / 2.41 | 0.655 / 0.587 |
| all | inbred_panel | 0.9 | 0.36 / 0.69 | 2.71 / 5.12 | 1.52 / 3.86 | 0.937 / 0.845 |
| all | hmdp | 0.9 | 0.07 / 0.16 | 1.47 / 1.87 | 1.30 / 2.05 | 0.940 / 0.928 |

Findings:

1. **Calibration.**
   * On a null phenotype, LOCO is calibrated (λ_GC 0.94-1.02; type-I 0.040-0.052 at 0.05 and
     0.0005-0.0012 at 1e-3), and so is the all-SNP K (0.92-0.98).
   * For SNPs on chromosomes without causal variants (`split`), LOCO is calibrated (λ 0.969-1.056).
     The all-SNP K is conservative there, increasingly with h2, and strongly so in inbred panels
     (λ 0.57, type-I at 1e-3 1.3e-5, 75x below nominal).
2. **λ_GC under LOCO measures linked signal, not calibration.** With the background on every chromosome
   (`all`), LOCO's λ over non-causal SNPs is 1.13-1.53 on the HMDP strains and 1.35-2.81 on the synthetic
   panel, rising with h2. The all-SNP K stays near 1 (0.95-1.01): its two biases cancel, absorbing the
   linked signal and deflating everything else. Only the unlinked outbred SNPs (family) stay at ~1.0 under
   LOCO. The iid null shows LOCO's test is calibrated where there is no signal. So:
   * do not read a LOCO λ_GC > 1 on a polygenic trait as confounding;
   * **never apply genomic control to a LOCO scan**: it would remove the linked signal that LOCO recovers;
   * check calibration on a null (permuted or simulated) phenotype, or on chromosomes without signal.
3. **LOCO gains power, a lot in inbred panels.**
   * At q2 = 0.05, the QTL's power at 1e-3 rises 0.29 → 0.43, 0.37 → 0.59 and 0.36 → 0.69 on the
     synthetic panel (`all`), and 0.07 → 0.16 on the HMDP strains at h2 0.9.
   * The background's causal SNPs' mean χ² rises 1.3-2.6x.
   * On outbred, unlinked SNPs the gain is small (+0.01 to +0.13 in mean −log10 p): only the tested SNP
     itself is in K.
   * Part of the gain in `all` comes with the general rise in LOCO statistics on the QTL's chromosome
     (finding 2). The `split` power numbers, where the null is calibrated, are the clean comparison, and
     show the same direction.
4. **The null h is a little lower with LOCO** for a polygenic trait (synthetic panel, h2 = 0.9: 0.85 vs
   0.94): `K_{-c}` misses chromosome c's background. Each chromosome gets its own h.
5. At h2 = 0 the two scans agree within noise.

## The README data (`just bench loco`)

`pylmm kinship` + `pylmm gwas --kinship` vs `pylmm gwas --loco` on the bundled 1219 mice × 98,740 SNPs on 19
autosomes (macOS arm64, 1 BLAS thread; wall time including start-up and I/O; three runs on a shared
machine):

| step | wall time (s) |
|---|---|
| standard: `pylmm kinship` | 5.4-8.3 |
| standard: `pylmm gwas --kinship` | 7.0-12.3 |
| standard: total | 12.4-20.6 |
| `pylmm gwas --loco` | 15.9-20.1 |
| (of which: building the 19 LOCO kinships, one pass) | 4.1-5.3 |

### Cost and memory as n grows

The extra cost of LOCO is **one eigendecomposition (O(n³)) and null fit per chromosome**, and that
dominates as n grows. Peak RSS and wall time (`/usr/bin/time -l`, default BLAS threads, after the review's
S-4 change; script `scratchpad/phase6-4/mem_run.sh` of the PR):

| data | `pylmm kinship` + `pylmm gwas` | `pylmm gwas --loco` | ratio |
|---|---|---|---|
| README, n = 1219, 19 chromosomes, 98,740 SNPs | 11.3 s; 231 MB / 303 MB | 14.2 s; 608 MB | 1.3x time, 2x memory |
| synthetic, n = 3000, 10 chromosomes, 1500 SNPs | 26.8 s; 309 MB / 504 MB | 198.8 s; 1009 MB | **7.4x** time, 2x memory |

* At n = 3000 one `eigh` of a 3000 × 3000 matrix takes ~20 s here, so 10 chromosomes cost ~200 s.
  Expect LOCO to cost roughly (number of chromosomes) × (one eigendecomposition) on top of the standard
  scan.
* Memory is about (C + 6) n² × 8 bytes plus the baseline, which `pylmm gwas --loco` logs before it starts
  (`pylmm.loco.loco_memory_bytes`). At n = 5000 with 20 chromosomes that is about 5 GB.
* The review measured 591 MB and 1.16 GB on the same two cases before the all-SNP K was dropped from the
  `--loco` path (another machine).

### λ_GC on the README data

**The fake phenotype** (h ≈ 0; null h 0.004 with all SNPs, 0.002-0.005 with LOCO):

| | λ_GC | SNPs at p < 1e-3 |
|---|---|---|
| standard | 1.050 | 18 |
| LOCO | 1.079 | 24 |

The top SNP is the same (mm37-7-19743810, p = 2.1e-4 in both), and the top 10 share 7 SNPs. Max
|Δ−log10 p| is 0.11 (median 0.004).

Why λ moves from 1.050 to 1.079, from the PR #30 review's analysis:
* The mean χ² ratio LOCO / all-SNP is 1.022, nearly uniform across chromosomes, and h is the same.
* Even at h = 0.004 the top eigenvalue of K is 175, so h·λ ≈ 0.7 along the top principal component.
  Removing the tested chromosome from K weakens that down-weighting: proximal contamination again, at
  tiny h.
* λ from one phenotype is noisy: per chromosome it ranges from 0.80 (chromosome 7) to 1.49
  (chromosome 19). Over 8 iid null phenotypes on these genotypes the review measured LOCO λ 1.013 ± 0.032
  and all-SNP λ 0.984 ± 0.021.

**Three seeded traits** on the same genotypes (one replicate each, seed 0, so each λ is a single noisy draw):

| trait | λ_GC standard | λ_GC LOCO | type-I 0.05 standard / LOCO |
|---|---|---|---|
| `split`: h2 = 0.5, 300 causal SNPs on odd autosomes; λ over the even (null) autosomes | 0.849 | 1.124 | 0.025 / 0.054 |
| `all`: h2 = 0.5, 300 causal SNPs on every autosome; λ over non-causal SNPs | 0.858 | **1.506** | 0.051 / 0.132 |
| `iid`: N(0, 1); λ over all SNPs | 1.096 | 1.121 | 0.047 / 0.050 |

The `all` row is the review's S-1 finding on real mouse genotypes: the review saw 1.537 ± 0.050 over 8
replicates with LOCO, vs 0.980 with the all-SNP K. That is linked signal, not miscalibration.

## Not covered

* The Bennett / Farber HMDP data of the replication (`docs/replication.md`) are not in the repository.
  The replication's hand-built LOCO moved −log10 p there by −0.6 to +1.2.
* Outbred data with real LD (the family panel's SNPs are unlinked); dosages; LOCO with `--refit`
  (supported, not simulated).
* An LD-score-style intercept, the calibration check for a polygenic LOCO scan, is not provided by pylmm.
