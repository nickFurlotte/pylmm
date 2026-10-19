---
name: custom-association-analysis
description: Write custom association analyses with the pylmm Python library (LMM, gwas, calculate_kinship, PlinkReader) when the pylmm gwas command line is not enough - scans on a subset of individuals, conditional analysis on a lead SNP, leave-one-chromosome-out (LOCO) kinship built by hand, many phenotypes sharing one eigendecomposition, SNP-by-covariate interaction, and inbred-strain panels with replicate animals. Use when asked for an ad-hoc or non-standard mixed-model analysis, a loop over phenotypes or chromosomes, or anything the CLI flags do not cover.
---

# Custom association analyses with the pylmm library

The building blocks (details in [docs/guide.md](../../docs/guide.md)):

* `calculate_kinship(W)` → K; `scipy.linalg.eigh(K)` → `kva, kve` (compute once, reuse).
* `LMM(Y, K, kva, kve, X0=X0)` then `.fit()` (null model, REML) and `.association(x)` (one SNP).
* `gwas(Y, X, K, kva, kve, X0=X0)` → lists of t and p (handles missing calls).
* `X0` is used as given: **always include `np.ones(n)`**.
* Genotypes from `PlinkReader` are standardised (`norm_genotype=True`); the `.bim` columns come from
  `plink_tools.read_bim` (same order as the SNP iterator).

The blocks below run in CI in order. They use a simulated cohort written as a PLINK file set: 400
individuals in 80 families, 3 chromosomes of 500 SNPs, a trait with one causal SNP on chromosome 2.

```python
import sys
import numpy as np
from scipy import linalg
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools
from pylmm.lmm import LMM, calculate_kinship, gwas
from pylmm.plink_io import PlinkReader

rng = np.random.default_rng(11)
n_fam, n_sib, m = 80, 5, 1500
p = rng.uniform(0.1, 0.5, m)
hap = rng.binomial(1, p, size=(n_fam, 4, m))                       # 4 parental haplotypes per family
G = np.concatenate([hap[:, rng.integers(0, 2)] + hap[:, 2 + rng.integers(0, 2)] for _ in range(n_sib)])
chrom = np.repeat(["1", "2", "3"], 500)
plink_tools.write_bed("cohort", G, chrom=chrom, pos=np.tile(np.arange(1, 501) * 1000, 3))
n = G.shape[0]
sex = rng.integers(0, 2, n).astype(float)
Gs = (G - G.mean(0)) / G.std(0)
K0 = calculate_kinship(G)
kva0, kve0 = linalg.eigh(K0)
u = kve0 @ (np.sqrt(np.clip(kva0, 0, None)) * rng.standard_normal(n)) * np.sqrt(0.4)
Y = 1 + 0.5 * sex + 0.35 * Gs[:, 700] + u + rng.normal(scale=np.sqrt(0.5), size=n)

with PlinkReader("cohort") as r:                                    # standardised genotypes
    W = np.column_stack([g for g, _ in r.iter_snps()])
bim = plink_tools.read_bim("cohort")
X0 = np.column_stack([np.ones(n), sex])
```

## Many phenotypes, one eigendecomposition

```python
K = calculate_kinship(W)
kva, kve = linalg.eigh(K)
phenos = {"trait": Y, "noise": rng.normal(size=n)}
results = {}
for name, y in phenos.items():
    ts, ps = gwas(y, W, K, kva, kve, X0=X0)
    results[name] = np.asarray(ps)
    print(name, "h2=%.2f" % LMM(y, K, kva, kve, X0=X0).fit()[0], "min p=%.1e" % np.nanmin(ps))
lead = int(np.nanargmin(results["trait"]))
print("lead SNP", bim["snp"][lead], "chr", bim["chrom"][lead])
```

`kva, kve` are ignored (recomputed) for a phenotype with missing values, since K is subset; for many
phenotypes with different missingness, group them by their missingness pattern.

## A subset of individuals

Subset `Y`, `X0` and both dimensions of `K`, then recompute the eigendecomposition (or let `LMM` do it).
Drop covariate columns that become constant (here `sex`):

```python
keep = sex == 1
ts_f, ps_f = gwas(Y[keep], W[keep], K[np.ix_(keep, keep)], X0=np.ones((keep.sum(), 1)))
print("females: n=%d, min p=%.1e" % (keep.sum(), np.nanmin(ps_f)))
```

Re-standardising `W` on the subset is not needed: t and p do not depend on the SNP's scale. With files,
`PlinkReader.subset(ids)` keeps individuals in `indivs`, `phenos`, `K` and the genotype vectors together.

## Conditional analysis on a lead SNP

Add the lead SNP to the covariates and rescan; a second signal that survives is independent of the lead:

```python
X0_cond = np.column_stack([X0, W[:, lead]])
ts_c, ps_c = gwas(Y, W, K, kva, kve, X0=X0_cond)
print("conditional on %s: min p=%.1e (lead itself: %s)" % (bim["snp"][lead], np.nanmin(ps_c), ps_c[lead]))
assert np.isnan(ps_c[lead])                # collinear with X0: untestable, as it should be
```

## LOCO: leave the tested chromosome out of K

A SNP in K partly explains itself (proximal contamination), which costs power. LOCO tests the SNPs of
chromosome c with a K built from the other chromosomes. Build the per-chromosome cross-products once:
`K_loco(c) = (Z Z' - Z_c Z_c') / (m - m_c)` with `Z` the standardised SNPs (here pylmm's `W`):

```python
ZZ = {c: W[:, chrom == c] @ W[:, chrom == c].T for c in np.unique(bim["chrom"])}
total, m_all = sum(ZZ.values()), W.shape[1]
p_loco = np.full(m_all, np.nan)
for c in ZZ:
    on_c = bim["chrom"] == c
    K_c = (total - ZZ[c]) / (m_all - on_c.sum())
    _, ps_c_loco = gwas(Y, W[:, on_c], K_c, X0=X0)
    p_loco[on_c] = ps_c_loco
print("lead SNP: p=%.1e with K from all SNPs, %.1e with LOCO" % (results["trait"][lead], p_loco[lead]))
```

For a file too large for memory, accumulate `ZZ[c]` while streaming SNP blocks with
`itertools.islice(reader.iter_snps(), block)`; alternatively split the BED per chromosome
(`plink_tools.subset_bed(prefix, out, bim["chrom"] != c)`), run `pylmm kinship` on each complement and
`pylmm gwas` on each chromosome. Each LOCO K needs its own eigendecomposition and gives its own null
`h` (slightly lower than with all SNPs). The CLI has no LOCO option yet (planned, Phase 6.4).

## SNP × covariate interaction

`association` tests the column it is given against `X0`; put the SNP's main effect in `X0` and test the
product:

```python
x = W[:, lead]
L_int = LMM(Y, K, kva, kve, X0=np.column_stack([X0, x]))
L_int.fit()
t_gxe, p_gxe = L_int.association(x * sex)
print("SNP x sex: t=%.2f p=%.2f" % (t_gxe, p_gxe))
```

## Inbred strains with replicate animals

In a panel of inbred strains, animals of one strain share a genotype. Two common set-ups:

* **Individual level** (EMMA-style): one row per animal; expand the strain kinship with the incidence
  matrix `Z` (animal × strain): `K_animals = Z K_strains Z'`. K is singular (rank = number of strains);
  pylmm handles that (eigenvalues below 1e-6 are floored).
* **Strain means**: average the phenotype per strain and fit one row per strain; h² is then on the
  strain-mean scale (higher than the individual-level h²).

```python
K_strains = K[:10, :10]                    # pretend the first 10 individuals are 10 strains
strain_of_animal = np.repeat(np.arange(10), 4)                     # 4 animals per strain
Z = np.eye(10)[strain_of_animal]
K_animals = Z @ K_strains @ Z.T
print(K_animals.shape, "rank", np.linalg.matrix_rank(K_animals))
```

Genotype rows must then be expanded the same way (`W_animals = Z @ W_strains`), and the IDs written for
the animals, not the strains.

## Checks for any custom loop

* Compare one SNP of your loop with `pylmm gwas` on the same inputs (they agree to ~1e-6 in p).
* Use `pylmm.lmm.snp_estimable(x, X0)` before `association` on arbitrary columns.
* Keep `reml` the same for the fit and the test (the default, True, does).
* For thresholds, λ_GC and effect directions see `interpret-gwas-results`; to check that a custom
  pipeline is calibrated, run it on null simulations (`simulate-lmm-data`).
