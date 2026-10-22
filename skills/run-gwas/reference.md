# run-gwas reference: phenotype QC, SNP QC, loci

Generic recipes around a `pylmm gwas` run. The blocks run in CI, in order, from the repository root
(`tests/test_docs.py`), on 3000 SNPs of the bundled mice and a simulated trait.

```python
import itertools, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
sys.path.insert(0, "skills/interpret-gwas-results/scripts")
import plink_tools, summarize_gwas
from pylmm.plink_io import PlinkReader

SRC = "data/snps.132k.clean.noX"
plink_tools.subset_bed(SRC, "geno", np.arange(3000))
with PlinkReader("geno") as r:
    ids = r.indivs
    x = next(itertools.islice(r.iter_snps(), 500, None))[0]      # standardised SNP 500
rng = np.random.default_rng(2)
n = len(ids)
y_raw = np.exp(0.5 * x + rng.normal(size=n))                       # a right-skewed, positive trait
y_raw[[3, 50]] = [400.0, 650.0]                                     # two gross outliers
y_raw[rng.random(n) < 0.02] = np.nan                                # some missing values
```

## Phenotype QC and transformation

Look first: the number of values, missing values, the range, skewness, and the extremes. Do it on the
individuals that will be analysed (after dropping those without genotypes or covariates), and per group
(sex, batch) if the groups differ in mean or spread.

```python
def describe(v):
    v = v[~np.isnan(v)]
    return "n=%d  mean=%.3g  sd=%.3g  min=%.3g  max=%.3g  skew=%.2f" % (
        len(v), v.mean(), v.std(), v.min(), v.max(), stats.skew(v))

print("raw:", describe(y_raw))
```

**Outliers.** Flag values far from the bulk with a robust rule, e.g. more than 5 robust SDs (median
absolute deviation × 1.4826) from the median, on the scale you will analyse; check them against the
source (unit errors, data entry) and set the confirmed ones to missing (`NA`) rather than editing them.
Report how many you removed.

```python
def robust_outliers(v, k=5.0):
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return np.abs(v - med) > k * mad                    # False for nan

y = y_raw.copy()
out = robust_outliers(np.log(y))
print("outliers on the log scale:", np.flatnonzero(out))
y[out] = np.nan
```

**Which transformation:**

* **None**, when the residuals are roughly symmetric: the LMM's Wald test tolerates mild non-normality.
* **log** (or `log1p` with zeros) for positive, right-skewed measurements whose variation is
  multiplicative (concentrations, counts, many weights and areas). Effects are then on the log scale
  (≈ relative change).
* **Rank-based inverse normal (RINT)**, `Φ⁻¹((rank - 0.5) / n)`, when no natural transformation removes
  heavy tails or skew. It caps the influence of outliers (which otherwise inflate false positives at SNPs
  with few minor-allele carriers), but effect sizes are in SD units of the transformed trait and are not
  comparable with untransformed analyses. Rank on the individuals analysed; with strong covariate
  effects, a common variant is to regress the covariates out first and rank the residuals.
* To replicate a published analysis, use **its** transformation, even if another one looks better.

```python
def rint(v):
    """Rank-based inverse normal transform; nan stays nan; ties get their average rank."""
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    out[ok] = stats.norm.ppf((stats.rankdata(v[ok]) - 0.5) / ok.sum())
    return out

y_log, y_rint = np.log(y), rint(y_raw)          # RINT is robust enough to skip the outlier step
for name, v in (("log", y_log), ("rint", y_rint)):
    print("%-4s %s" % (name, describe(v)))
plink_tools.write_pheno("geno.pheno", ids, np.column_stack([y_raw, y_log, y_rint]), ["raw", "log", "rint"])
```

```bash
pylmm kinship --bfile geno -o geno.kin -v
pylmm gwas --bfile geno --kinship geno.kin --pheno geno.pheno --pheno-name log -o log.tsv.gz -v
```

## SNP QC

Typical filters: minor-allele frequency (MAF) and per-SNP missingness (call rate); also per-sample
missingness, and Hardy–Weinberg equilibrium in outbred samples only (inbred strains are homozygous by
design). Thresholds depend on n: in a few dozen strains a MAF of 0.05 is one or two strains, so think in
minor-allele *counts* of distinct strains, not only frequencies.

In numpy, from the raw dosages (`norm_genotype=False` gives the A2 fraction, nan = missing), into an
index list for `plink_tools.subset_bed`:

```python
maf, miss = [], []
with PlinkReader("geno", norm_genotype=False) as r:
    for g, _ in r:
        called = ~np.isnan(g)
        f = g[called].mean() if called.any() else 0.0
        maf.append(min(f, 1 - f))
        miss.append(1 - called.mean())
maf, miss = np.array(maf), np.array(miss)
keep = np.flatnonzero((maf >= 0.05) & (miss <= 0.10))
print("SNP QC: %d of %d kept (MAF >= 0.05, missingness <= 10%%)" % (len(keep), len(maf)))
plink_tools.subset_bed("geno", "geno_qc", keep)
```

In inbred panels with replicate animals, compute the MAF over distinct strains (one animal per strain)
if the question is how many strains carry the minor allele. The plink2 equivalent:

```bash skip-ci: needs plink2
plink2 --bfile geno --maf 0.05 --geno 0.10 --make-bed --out geno_qc
```

Build the kinship from the QC'd SNPs as well (`pylmm kinship --bfile geno_qc`).

## Loci: from SNPs to regions

A scan reports SNPs; results are loci. `skills/interpret-gwas-results/scripts/summarize_gwas.py` defines
them by distance (`--clump-kb KB`: the best SNP, then every SNP within KB of it on its chromosome, then
the best SNP left), with the per-allele effect of each lead (see `interpret-gwas-results` sections 2
and 6):

```bash
pylmm gwas --bfile geno --kinship geno.kin --pheno geno.pheno --pheno-name rint -o rint.tsv.gz -v
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py rint.tsv.gz --clump-kb 2000 --clump-p 1e-5 --top 5
```

An LD-based interval around a lead SNP: the SNPs within the window whose genotypes correlate with the
lead's at r² ≥ 0.8 (a common choice; inbred panels often need a stricter one, as LD is long):

```python
res = summarize_gwas.read_results("rint.tsv.gz")
bim = summarize_gwas.positions(res)                              # chrom, pos, alleles of the table
lead = summarize_gwas.rank(res)[0]
with PlinkReader("geno") as r:                                   # standardised, missing -> 0
    W = np.column_stack([g for g, _ in r])
window = np.flatnonzero((bim["chrom"] == bim["chrom"][lead]) & (np.abs(bim["pos"] - bim["pos"][lead]) <= 2_000_000))
r2 = np.array([np.corrcoef(W[:, lead], W[:, j])[0, 1] ** 2 for j in window])
interval = window[r2 >= 0.8]
print("lead %s chr%s:%d; r2 >= 0.8 interval %d-%d (%d SNPs)" % (
    res["id"][lead], bim["chrom"][lead], bim["pos"][lead],
    bim["pos"][interval].min(), bim["pos"][interval].max(), len(interval)))
lo, hi = bim["pos"][interval].min(), bim["pos"][interval].max()
assert lo <= bim["pos"][500] <= hi            # the planted SNP 500 lies in the lead's interval
```

The lead SNP need not be the causal one: here the planted SNP 500 is ~1.3 Mb from the lead, inside its
interval. Compare loci, not lead SNPs, between analyses.
