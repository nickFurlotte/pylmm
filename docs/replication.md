# Validated against published results: two HMDP mouse GWAS

In October 2026 a fresh agent, working only from [AGENTS.md](../AGENTS.md), [guide.md](guide.md) and the
[skills](../skills/), reanalysed two published GWAS of the Hybrid Mouse Diversity Panel with pylmm. It read
no pylmm source and found no numerical bug. It used the original phenotypes (paper supplement and Mouse
Phenome Database exports), the bundled HMDP genotypes ([data/README.md](../data/README.md)) and their
build-37 marker map, which it recovered and validated during the run. Intervals (published peak ± 1.3 Mb,
build 37) and thresholds were written down before the scans.

## Results

**Farber et al. 2011** (*PLoS Genet* 7:e1002038), bone mineral density, 94 of 97 strains, 831–844
animals, 105,130 SNPs. **Partially replicated.** All five published trait–locus pairs have a nearby
signal (the best SNP in each published interval; an interval always has a best SNP, so this alone is not
a replication), three at the published position (to the 0.01 Mb the paper reports). Significance is
**systematically lower, by 0.47 in −log10 p (median; range 0.13–0.58)**; **1/5 pass Bonferroni**
(4.76e-7) and **3/5 the paper's threshold** (4.1e-6). Closest method (IBS-style K, per-SNP refit):

| trait, published lead | published chr:Mb | published −log10 p | ours chr:bp (b37) | ours −log10 p |
|---|---|---:|---|---:|
| TBMD, rs32149600 | 7:20.01 | 6.60 | 7:19,742,350 | 6.13 |
| TBMD, rs29131970 | 12:3.50 | 5.77 | 12:3,501,831 | 5.26 |
| TBMD, rs33294019 | 17:43.91 | 7.35 | 17:43,910,500 | 6.77 |
| FBMD, rs32149600 | 7:20.01 | 5.59 | 7:19,742,350 | 5.46 |
| SBMD, rs26963925 | 11:56.77 | 5.60 | 11:56,770,801 | 5.26 |

Likely causes of the lower p-values (none tested on its own): three strains without verified genotypes
(BXHA1, BXHB2, BXHE1), a different SNP set and imputation (105,130 vs the paper's 108,064 SNPs), and an
approximate reconstruction of EMMA's IBS kinship.

**Bennett et al. 2010** (*Genome Res* 20:281), HDL cholesterol, 95 of 99 strains, 660 animals, 105,249
SNPs. **Replicated: the lead SNP is the published one**: 1:172,441,271, the *Nos1ap* peak at 172.4 Mb, top of the
genome in every analysis mode. −log10 p **11.5** with the per-SNP refit vs **9.4** published (7.9–9.0 in
the null-fit modes; Bonferroni 6.3). Individual-level SNP h² 0.80 [0.75, 0.85] (the paper reports none).

## Method choices

* Individual animals as the unit (`K_animals = Z K_strains Z'`), raw phenotype, intercept only, REML.
* SNP QC over strains: autosomes, MAF ≥ 0.05, missing < 0.2; missing calls mean-imputed over strains.
* Four scans per trait: standardized K (`calculate_kinship`) and IBS-style K, null fit only; LOCO; and
  IBS-style K with a per-SNP REML refit (`fit(X=x)`, `association(x)`), closest to EMMA. The refit and
  the IBS kinship were the two levers that moved p-values toward the papers; see the
  [comparison worksheet](../skills/interpret-gwas-results/comparison-worksheet.md).
* Strain aliases from exact joins (animal ID and trait values) between sources; duplicate strain columns
  audited for genotype identity before use (`inbred-strain-panels` skill).
* Cross-check with the command line on a per-animal BED: `pylmm gwas` matches the library to 8e-13 in
  −log10 p where the inputs are identical, and differs by up to 0.28 on SNPs with missing calls, which the
  CLI imputes over animals instead of strains.

## Caveats

* A reanalysis of the original observations, not an independent cohort.
* The phenotype exports may not hold exactly the animals analysed in 2010–2011; 3 (Farber) and 4
  (Bennett) strains lack verified genotypes; the SNP sets and the kinship differ from the papers'.
* Effect direction is **not assessed**: neither paper gives a signed effect for a named allele.
* Heritability is estimated, not compared: neither paper reports a numerical h².
* Bennett's published p = 4.4e-10 (Table 1, an image) was not re-verified in the second session, and its
  secondary loci were not assessed (Table 1 and Supplemental Table 4 were not available).
* Data acquisition, strain repair and method matching were iterative (the intervals and thresholds were
  not); every mode tried is reported. A third study (Parker et al. 2016, CFW mice) was not assessed: its
  Dryad download returned HTTP 403.
* pylmm's default (standardized K, variance fitted once) gave −log10 p 1.1–1.7 (Farber) and 3.6
  (Bennett) below the IBS-style refit, i.e. below the papers for Farber and for Bennett (7.9 vs 9.4),
  while the refit overshot Bennett (11.5). Use `--refit` / `refit=True` when matching an EMMA or
  GEMMA-exact analysis, and report which mode you used.

Planned from these findings: a named IBS kinship option, strain-aware expansion and imputation,
a LOCO flag for the command line, marker metadata through the EMMA path, and a machine-readable null-fit
summary.
