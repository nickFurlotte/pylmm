# Worksheet: comparing a pylmm scan with a published GWAS

Use this before you say a published association "replicated" or "did not replicate". Fill in one copy per
study, **before** looking at your p-values for the intervals and thresholds, and keep it with the
analysis. Each item is something that changed −log10 p by about 0.3–3.5 units in the [replication](../../docs/replication.md)
of two HMDP mouse GWAS ([docs/replication.md](../../docs/replication.md)); together they explain most
"why is my p-value different".

The blocks below run in CI, in order, on a small synthetic strain panel.

## The template

```text
STUDY                <paper, year, trait>; data source <supplement / MPD / repository + version>
OBSERVATION UNIT     paper: animals | strain means      ours: animals | strain means
                     replicates per strain <min-max>, harmonic mean r <r_h>
PHENOTYPE            units <..>; transform paper <none/log/rank>  ours <..>; covariates paper <..>  ours <..>
                     sex / diet / age subset <..>; outlier rule <..>
KINSHIP              paper: <formula: IBS / standardized (GRM) / centred IBS / LOCO>, scale <Gower? diag?>
                     ours:  <formula>, scale trace(PKP)/(n-1) = <..>
VARIANCE FIT         paper: per-SNP refit (EMMA, GEMMA -lmm exact) | null only (EMMAX / P3D / FaST-LMM)
                     ours:  per-SNP refit (gwas(refit=True) / --refit) | null only (default)
                     REML | ML; test: Wald t | F | LRT | score
GENOME BUILD         paper <mm9/NCBI37 ...>  ours <..>   lifted over? <tool, chain>
EFFECT ALLELE        paper names <allele / not given>  ours counts <A2 / allele coded 1>  -> same? flip?
SNP SET / QC         paper: <n SNPs>, MAF <..>, missing <..>, source <..>
                     ours:  <n SNPs>, MAF <..> over <strains / animals>, missing <..>, imputed over <..>
SAMPLES              paper <n strains, n animals>  ours <..>; strains missing from ours <list, why>
THRESHOLDS           paper <value, how derived>; ours: Bonferroni 0.05/<m> = <..>, also report paper's
INTERVALS            pre-registered: <published peak +- X Mb (why X)>, on build <..>
VERDICT RULES        replicated = positional AND significance: the best SNP in the pre-registered interval
                     passes <threshold>. Positional agreement alone (an interval always has a best SNP)
                     is necessary but not sufficient; report it as "nearby signal".
                     Label: replicated (all loci pass) | partially replicated (some pass; give k/n
                     per threshold) | not replicated | not assessed
                     significance: |our -log10 p - published| <= <margin> (stated before the scan)
                     direction: compared only for the same named allele, else "not assessed"
RESULT               per locus: our best SNP, chr:bp, -log10 p, threshold met?, difference, verdict
                     summary: "<k>/<n> pass <threshold 1>, <j>/<n> pass <threshold 2>" - never just
                     "n/n found in their intervals"
```

## The items, and what pylmm does

**Kinship formula and scale.** EMMA-era mouse papers (Farber 2011, Bennett 2010) used an IBS kinship;
pylmm's `calculate_kinship` is the standardized (GRM-style) one. The *scale* changes h² but no test
statistic (`estimate-heritability` section 3); the *formula* changes the weighting and so the p-values.
An EMMA-style homozygous IBS kinship for 0/1 strain genotypes, centred and Gower-scaled so that it is on
the same scale as `calculate_kinship(G, center=True)` (a named IBS option is planned):

```python
import numpy as np
from pylmm.lmm import LMM, calculate_kinship, gwas

rng = np.random.default_rng(1)
n_strains, m = 60, 800
founders = rng.integers(0, 2, size=(8, m)).astype(float)
G = founders[rng.integers(0, 8, size=(n_strains, m)), np.arange(m)]   # strains x SNPs, 0/1, complete
G = G[:, (G.mean(0) > 0.05) & (G.mean(0) < 0.95)]

def ibs_kinship(G01):
    """Homozygous IBS share (1 + A A'/m) / 2 with A = 2G - 1, centred and Gower-scaled.

    G01 must be complete (mean-impute over strains first); with missing calls this is not the
    pairwise-complete IBS some tools compute.
    """
    A = 2 * G01 - 1
    K = (1 + A @ A.T / G01.shape[1]) / 2
    P = np.eye(len(K)) - 1.0 / len(K)
    K = P @ K @ P
    return K * (len(K) - 1) / np.trace(K)

K_ibs = ibs_kinship(G)
K_std = calculate_kinship(G, center=True)
for K in (K_ibs, K_std):
    assert np.isclose(np.trace(K) / (n_strains - 1), 1.0)   # Gower scale: trace(PKP)/(n-1) = 1
print("max |K_ibs - K_std| = %.2f" % np.abs(K_ibs - K_std).max())
```

**Observation unit.** A scan of animals (one row per animal, `K_animals = Z K_strains Z'`) and a scan of
strain means are different models with different h² and p-values (`estimate-heritability` section 7,
`inbred-strain-panels`). Match the paper's unit; if you cannot, say so.

**Variance fit: per-SNP refit vs null only.** EMMA and GEMMA's exact test refit the variance components
for every SNP; EMMAX, P3D and pylmm's default fit them once without SNPs. The refit is the closer match
to an EMMA paper (`gwas(..., refit=True)`, `pylmm gwas --refit`, or `fit(X=x)` then `association(x)`):

```python
x = G[:, 0]
y = 0.8 * (x - x.mean()) / x.std() + rng.multivariate_normal(np.zeros(n_strains), K_ibs)
y += rng.normal(scale=0.5, size=n_strains)
_, p_null = gwas(y, G[:, :30], K_ibs)                # h fitted once, without SNPs
_, p_refit = gwas(y, G[:, :30], K_ibs, refit=True)   # h refitted for each SNP
L = LMM(y, K_ibs)
h_null = L.fit()[0]
h_snp = L.fit(X=x[:, None])[0]                       # refit with the SNP in the model ...
t1, p1 = L.association(x)                            # ... then test it: the same as refit=True
assert np.isclose(p1, p_refit[0]) and p_refit[0] != p_null[0]
print("SNP 0: h %.2f without it, %.2f with it; -log10 p null-only %.2f, per-SNP refit %.2f"
      % (h_null, h_snp, -np.log10(p_null[0]), -np.log10(p_refit[0])))
```

**Phenotype transform, covariates, subset.** Raw vs log vs rank-normal can move a peak (log10 HDL moved
the Bennett lead from 172.4 to 173.1 Mb inside the same interval). Use the paper's, or report both.

**Genome build.** Compare positions on one build; the HMDP map is build 37 (mm9), most current mouse
resources are GRCm38/39. Lift over before matching; never match by rsID alone.

**Named effect allele.** A sign is comparable only for the same named base. pylmm counts A2 (BED) or the
allele coded 1 (EMMA; `allele1` of the HMDP map). If the paper names no allele, direction is "not
assessed", not "agrees".

**SNP set and QC.** Different SNP sources, MAF and missingness filters, and imputation (over strains vs
over animals, `inbred-strain-panels` section 5) change the best SNP in a region and its p-value.

**Thresholds.** Report Bonferroni for your SNP set and the paper's threshold (often permutation or
bootstrap based, `SKILL.md` section 3). Do not swap in a threshold after seeing the results.

## Judging "replicated"

Decide the rule before the scan, and report the numbers, not only the verdict:

* **Positional replication**: the best SNP inside a pre-registered interval around the published peak
  (e.g. ± the published LD interval, or ± 1.3 Mb as in the replication) passes the stated threshold, on the
  same build. Exact equality of the lead SNP is not required.
* **Significance within a stated margin**: |our −log10 p − published −log10 p| ≤ a margin you state in
  advance (e.g. 1 unit). A locus at −log10 p 5.26 against a threshold of 5.39 is *not* a failure to
  replicate when the paper had 5.77: treat a near-threshold flip as "same signal, slightly weaker", and say
  so, rather than as a binary miss.
* **Direction**: only for the same named allele.

```python
def judge(best_logp, published_logp, threshold_logp, margin=1.0):
    """Verdict for one locus, from -log10 p values (best SNP in the pre-registered interval)."""
    within = abs(best_logp - published_logp) <= margin
    passes = best_logp >= threshold_logp
    if passes:
        return "replicated (passes threshold)" + ("" if within else ", outside margin")
    return "same signal within margin, below threshold" if within else "not replicated"

# Farber 2011, TBMD at Chr12 (Asxl2): ours 5.256 vs published 5.77, paper threshold -log10(4.1e-6)
print(judge(5.256, 5.77, -np.log10(4.1e-6)))
assert judge(5.256, 5.77, 5.387) == "same signal within margin, below threshold"
assert judge(6.772, 7.35, 6.32).startswith("replicated")
```

## What the replication found

With the paper's phenotypes and the bundled HMDP genotypes, **an IBS-style K with a per-SNP REML refit
was the closest to the EMMA papers**: Farber 2011, 4.0–5.0 → 5.3–6.8 in −log10 p from the standardized
null-only scan to the IBS refit across the five published loci (published 5.6–7.35); Bennett 2010 HDL,
7.88 (standardized, null only) and 7.95 (IBS, null only) → 11.48 (IBS, refit), published 9.36, same lead
SNP in every mode. The variance fit (IBS K, null-only → refit: +0.7 to +3.5) and the kinship
formula (null-only, standardized → IBS: +0.07 to +0.41) were the steady levers; LOCO moved −log10 p by −0.6 to +1.2 depending on
the locus, and the imputation unit by up to 0.28. Details: [docs/replication.md](../../docs/replication.md).
