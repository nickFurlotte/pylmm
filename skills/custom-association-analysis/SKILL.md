---
name: custom-association-analysis
description: Write custom association analyses with the pylmm Python library (LMM, gwas, calculate_kinship, PlinkReader) when the pylmm gwas command line is not enough - scans on a subset of individuals, conditional analysis on a lead SNP, leave-one-chromosome-out (LOCO) scans with pylmm.loco (and how they work by hand), many phenotypes sharing one eigendecomposition, SNP-by-covariate interaction, and scans of inbred-strain panels with replicate animals. Use when asked for an ad-hoc or non-standard mixed-model analysis, a loop over phenotypes or chromosomes, or anything the CLI flags do not cover.
---

# Custom association analyses with the pylmm library

The building blocks (details in [docs/guide.md](../../docs/guide.md)):

* `calculate_kinship(G_raw)` → K, from **raw** genotypes (`norm_genotype=False`; then it is `pylmm kinship`'s K); `scipy.linalg.eigh(K)` → `kva, kve` (compute once, reuse).
* `LMM(Y, K, kva, kve, X0=X0)` then `.fit()` (null model, REML) and `.association(x)` (one SNP), or
  `.association_batch(X)` (many complete SNPs at once: arrays `t, p, beta, beta_var`; much faster in a
  loop; check `snp_estimable` first, as for `association`).
* `gwas(Y, X, K, kva, kve, X0=X0)` → lists of t and p (handles missing calls; batched by default,
  `batch=False` for one SNP at a time).
* `X0` is used as given: **always include `np.ones(n)`**.
* Genotypes from `PlinkReader` are standardised (`norm_genotype=True`); the `.bim` columns are
  `reader.variants` (`chrom`, `snp`, `cm`, `bp`, `a1`, `a2` arrays, same order as the SNPs).
* `reader.iter_blocks(block_size)` yields `(G, variants)` blocks: `G` is `(n, b)`, bit for bit the
  per-SNP vectors, decoded a block at a time (faster for BED). `pylmm.genotypes.ArraySource(G, ...)`
  gives an in-memory matrix the same interface.

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
    W = np.hstack([G_block for G_block, _ in r.iter_blocks()])
    bim = r.variants                                                # chrom, snp, cm, bp, a1, a2
with PlinkReader("cohort", norm_genotype=False) as r:               # raw 0/0.5/1, for the kinship
    G_raw = np.hstack([G_block for G_block, _ in r.iter_blocks()])
X0 = np.column_stack([np.ones(n), sex])
```

## Many phenotypes, one eigendecomposition

```python
K = calculate_kinship(G_raw)          # raw genotypes: reader-standardised W would be standardised twice
kva, kve = linalg.eigh(K)
phenos = {"trait": Y, "noise": rng.normal(size=n)}
results = {}
for name, y in phenos.items():
    ts, ps = gwas(y, W, K, kva, kve, X0=X0)
    results[name] = np.asarray(ps)
    print(name, "h2=%.2f" % LMM(y, K, kva, kve, X0=X0).fit()[0], "min p=%.1e" % np.nanmin(ps))
lead = int(np.nanargmin(results["trait"]))
print("lead SNP", bim.snp[lead], "chr", bim.chrom[lead])
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
print("conditional on %s: min p=%.1e (lead itself: %s)" % (bim.snp[lead], np.nanmin(ps_c), ps_c[lead]))
assert np.isnan(ps_c[lead])                # collinear with X0: untestable, as it should be
```

## LOCO: leave the tested chromosome out of K

A SNP in K partly explains itself (proximal contamination): power drops and, with a polygenic trait,
the statistics are deflated (λ_GC well below 1 in inbred panels; `sims/studies/s8_loco.md`). LOCO
tests the SNPs of chromosome c with `K_{-c}`, the kinship of the other chromosomes. On the command line
it is `pylmm gwas --loco` (`run-gwas`). In the library, `pylmm.loco.LocoKinship` builds every `K_{-c}`
in one pass over the **raw** genotypes, and `gwas_loco` scans with them (a null fit per chromosome):

```python
from pylmm.loco import LocoKinship, gwas_loco

loco = LocoKinship.from_genotypes(G_raw, bim.chrom)          # reuse it for every phenotype
res = gwas_loco(Y, W, bim.chrom, kinships=loco, X0=X0)       # (t_stats, p_values, h per chromosome)
p_loco = np.asarray(res.p_values)
print("null h per chromosome:", {c: round(v, 3) for c, v in res.h.items()})
print("lead SNP: p=%.1e with K from all SNPs, %.1e with LOCO" % (results["trait"][lead], p_loco[lead]))
```

`gwas_loco(Y, X, chroms)` without `kinships` builds them from `X` itself (then pass raw genotypes);
`kind=` picks `"centered"` or `"ibs"` as for `calculate_kinship`. `loco.kinship(c)` returns `K_{-c}`
(`loco.kinship()`: all SNPs) for your own loops, e.g. with `LMM` and `association_batch`.

What it computes, by hand: each kinship is a sum of per-SNP terms divided by the number of SNPs used,
so `K_{-c} = (S - S_c) / (m - m_c)` with `S_c = Z_c Z_c'` over chromosome c's standardised SNPs. The
library keeps one `S_c` per chromosome (memory: one n × n matrix per chromosome) and gives the same K as
`calculate_kinship` of the complementary SNPs, to rounding:

```python
c = bim.chrom[lead]
on_c = bim.chrom == c
K_c = calculate_kinship(G_raw[:, ~on_c])                     # the direct way, one chromosome
assert np.allclose(loco.kinship(c), K_c, rtol=0, atol=1e-12)
_, ps_c = gwas(Y, W[:, on_c], K_c, X0=X0)
assert np.allclose(ps_c, p_loco[on_c], rtol=1e-6)
```

Each `K_{-c}` needs its own eigendecomposition and gives its own null `h` (slightly lower than with all
SNPs when the trait is polygenic). For files too large for memory, feed `LocoKinship.add_block(G,
v.chrom)` from `PlinkReader(..., norm_genotype=False).iter_blocks()`.

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

`association_batch` tests many columns against one fitted model at once. Every column shares `L_int`'s
`X0`, which holds the lead SNP's main effect, not each SNP's own; a proper interaction scan needs one
model per SNP:

```python
res = L_int.association_batch(W[:, :50] * sex[:, None])   # arrays of 50: t, p, beta, beta_var
assert np.allclose(res.t[0], L_int.association(W[:, 0] * sex)[0], rtol=1e-9)
```

## Inbred strains with replicate animals

The set-up (individual level with `K_animals = Z K_strains Z'` vs strain means, the strain-mean
heritability, what to expect in small panels) is in one place: `estimate-heritability` section 7. For a
scan at the individual level, expand the genotypes like K, by row indexing (`W_animals =
W_strains[strain_of_animal]`; `Z @ W_strains` would spread a missing call to every animal), and write the
IDs of the animals, not the strains. With few distinct strains, prefer LOCO (above): a K from the tested
SNPs absorbs much of the signal (`simulate-lmm-data` section 6). Preparing the data (strain aliases and
duplicate columns, imputing over strains rather than animals, PLINK-safe strain IDs, the EMMA strain-mean
path): the `inbred-strain-panels` skill.

## Checks for any custom loop

* Compare one SNP of your loop with `pylmm gwas` on the same inputs (they agree to ~1e-6 in p).
* Use `pylmm.lmm.snp_estimable(x, X0)` before `association` on arbitrary columns.
* Keep `reml` the same for the fit and the test (the default, True, does).
* For thresholds, λ_GC and effect directions see `interpret-gwas-results`; to check that a custom
  pipeline is calibrated, run it on null simulations (`simulate-lmm-data`).
