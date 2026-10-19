---
name: interpret-gwas-results
description: Interpret pylmm GWAS output - what SNP_ID, BETA, BETA_SD, F_STAT and P_VALUE mean, which allele and scale BETA refers to and how to convert it to per-allele effects comparable with plink2, GEMMA, GCTA, REGENIE or BOLT-LMM, genome-wide significance thresholds (Bonferroni, effective number of tests, a parametric-bootstrap threshold for related samples), genomic control lambda_GC and QQ plots, comparing hits with published loci, and when not to trust a result (near-singular kinship, h2 at the 0 or 0.99 boundary, small n, rare genotypes, inbred panels). Use when reading, summarising, thresholding or comparing pylmm association results.
---

# Interpret pylmm GWAS results

The blocks below run in CI, in order, from the repository root. `scripts/summarize_gwas.py` (this
skill) prints counts, λ_GC, the Bonferroni threshold and the top SNPs of an output file (ranked by p, ties
such as p = 0.0 by |t|). Options: `--bfile PREFIX` joins chromosome, position and alleles from the
`.bim`; `--per-allele` adds per-copy effects (section 2); `--clump-kb KB` / `--clump-snps N` group hits
into loci (section 6); `--plot out.png` draws a Manhattan plot (x = base-pair position with `--bfile`) and
a QQ plot (needs matplotlib).

## 1. The output file

Tab-separated, one line per `.bim` SNP, in `.bim` order:

| column | meaning | careful |
|---|---|---|
| `SNP_ID` | `.bim` column 2 | join chromosome/position from the `.bim` (same order) |
| `BETA` | effect of the **A2** allele (`.bim` column 6), **per SD** of the standardised genotype | not per allele; A2 = REF for plink2 files |
| `BETA_SD` | **standard error** of BETA | not an SD |
| `F_STAT` | signed **Wald t** = BETA / BETA_SD | not an F (F = t²) |
| `P_VALUE` | two-sided, t(n - q - 1) | exactly 0.0 if it underflows (very large t); then rank by abs(F_STAT) |

All four values are `nan` for an untestable SNP (constant, collinear with covariates, all missing).

```python
import sys
import numpy as np
from scipy import linalg, stats
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
sys.path.insert(0, "skills/interpret-gwas-results/scripts")
import plink_tools, summarize_gwas
from pylmm.lmm import LMM, calculate_kinship, gwas

# a small simulated scan: 300 unrelated people, 1000 SNPs on 2 chromosomes (10 kb apart),
# SNP 10 causal (A2 allele raises y)
rng = np.random.default_rng(1)
G = rng.binomial(2, rng.uniform(0.05, 0.5, 1000), size=(300, 1000)).astype(float)
y = 0.5 * G[:, 10] + rng.normal(size=300)
iids = ["i%d" % i for i in range(300)]
plink_tools.write_bed("scan", G, iids=iids, chrom=np.repeat(["1", "2"], 500),
                      pos=np.tile(np.arange(1, 501) * 10_000, 2))      # G = copies of A2
plink_tools.write_pheno("scan.pheno", [(i, i) for i in iids], y, ["y"])
```

```bash
pylmm kinship --bfile scan -o scan.kin -v
pylmm gwas --bfile scan --kinship scan.kin --pheno scan.pheno -o scan.tsv -v
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv --bfile scan --top 3
```

## 2. Effect allele and scale

pylmm standardises each SNP, so `BETA` is per SD of the A2-allele dosage. Per copy of A2:
`BETA / sd`, with `sd` the SD of the 0/1/2 dosage over the called individuals; per copy of A1 (the
counted allele of plink2 `--glm`, GEMMA, GCTA, REGENIE, BOLT-LMM): `-BETA / sd`. Same for the SE. t and p
need no conversion (t flips sign with the allele).

```python
res = summarize_gwas.read_results("scan.tsv")
sd = np.nanstd(G, axis=0)                                          # SD of the A2 dosage
beta_a2 = res["BETA"] / sd
se_a2 = res["BETA_SD"] / sd
print("SNP 10: per-SD %.3f, per A2 copy %.3f (true 0.5), SE %.3f" % (res["BETA"][10], beta_a2[10], se_a2[10]))
assert abs(beta_a2[10] - 0.5) < 3 * se_a2[10]
```

`summarize_gwas.py --bfile PREFIX --per-allele` does this conversion for you: it reads the raw dosages
from `PREFIX.bed` and reports, per hit, the effect and SE per copy of A2 and the effect per copy of A1
(the sign flipped), with both allele names. The SD is taken over all called individuals of the `.fam`, as
pylmm standardises; it is approximate if the kinship dropped individuals, and wrong for scans run with
`--remove-missing-genotypes` (whose BETA is on another scale).

```bash
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv --bfile scan --per-allele --top 3
```

```python
sd_bed = summarize_gwas.dosage_sd("scan")                         # the same SDs, from the BED
assert np.allclose(sd_bed, sd)
s = summarize_gwas.summarize(res, summarize_gwas.read_bim("scan"), top=1, sd=sd_bed)
print(s["top"][0])
assert s["top"][0]["index"] == 10 and np.isclose(s["top"][0]["beta_a2"], beta_a2[10])
assert np.isclose(s["top"][0]["beta_a1"], -beta_a2[10])
```

| tool | effect allele in its output | scale |
|---|---|---|
| pylmm (today) | A2 (`.bim` col 6; REF for plink2-written files) | per SD |
| plink2 `--glm` | `A1` column (ALT by default) | per allele |
| GEMMA `-lmm` | `allele1`, usually `.bim` A1 (not verified here: check the GEMMA manual for which allele `beta` refers to) | per allele |
| GCTA `--mlma`, fastGWA | `A1` | per allele |
| REGENIE | `ALLELE1` (ALT) | per allele |
| BOLT-LMM | `ALLELE1` | per allele |
| SAIGE | `Allele2` | per allele |

plink 1.9 without `--keep-allele-order` makes A1 the minor allele, so "A2" can be the major allele and the
sign can differ between datasets. Compare alleles by name, never by position.

## 3. Significance thresholds

* **Bonferroni**: `0.05 / m_tested` (nan SNPs excluded). Conservative with LD.
* **5e-8** is the human genome-wide convention (~1M independent common variants). It does not transfer to
  other species, panels or SNP densities.
* **Effective number of tests**: from the eigenvalues of the SNP correlation matrix (Li & Ji 2005); then
  `0.05 / M_eff`. Compute it per chromosome or LD block and sum:

```python
R = np.corrcoef(G[:, :200], rowvar=False)              # SNP correlation (200 SNPs here)
lam = np.abs(linalg.eigvalsh(R))
m_eff = np.sum((lam >= 1) + (lam - np.floor(lam)))
print("M_eff of 200 independent SNPs: %.0f" % m_eff)   # < 200: with n = 300, sampling noise in R
                                                    # spreads the eigenvalues; M_eff is a rough guide
```

* **Parametric bootstrap** (the honest threshold for related or structured samples): simulate phenotypes
  from the fitted *null* model, scan them, and take the 5% quantile of the minimum p. (Permuting `y` is
  wrong under relatedness: it breaks the covariance the LMM models.)

```python
K = np.loadtxt("scan.kin")
kva, kve = linalg.eigh(K)
null = LMM(y, K, kva, kve)
h, beta0, s2, _ = null.fit()
min_p = []
for _ in range(20):                                        # use 1000+ for a real threshold
    u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(300)) * np.sqrt(h * s2)
    y_null = beta0[0, 0] + u + rng.normal(scale=np.sqrt((1 - h) * s2), size=300)
    min_p.append(np.nanmin(gwas(y_null, G, K, kva, kve)[1]))
print("bootstrap 5%% min-p threshold ~ %.1e (Bonferroni %.1e)" % (np.quantile(min_p, 0.05), 0.05 / 1000))
```

* **FDR** (Benjamini–Hochberg) for exploratory follow-up, not for claiming loci.

## 4. Genomic control and QQ plots

λ_GC = median(χ²₁ of the p-values) / 0.455. With a well-specified LMM it is ~1 for null SNPs.

```python
lam_gc = summarize_gwas.lambda_gc(res["P_VALUE"])
print("lambda_GC = %.3f" % lam_gc)
assert 0.8 < lam_gc < 1.25
```

* λ > 1.1 with an LMM: missing covariates (batch, sex), a kinship that does not match the samples (wrong
  order: check the `.id` file), or a K from too few or the wrong SNPs. Look at the QQ plot: a lifted whole
  curve is inflation, a tail-only departure is signal. (The rule that λ grows with n·h²/M under
  polygenicity is the linear-regression, no-LMM result. Under an LMM whose K includes the tested SNPs,
  polygenic signal tends to *deflate* the test statistics instead, because K absorbs it; LOCO is the
  remedy, see `custom-association-analysis`.)
* λ < 1: K built from the tested SNPs absorbing signal (proximal contamination; LOCO helps), or h² = 0 fits
  (mildly conservative, Phase 4).
* **Few effective tests** (inbred panels, small scans, one chromosome): λ from one scan is noise (Phase 4:
  5–95% range 0.7–1.5 under the null on the bundled mice). Do not GC-correct LMM results by default.

## 5. When to distrust a result

* **h at a boundary**: `null model: heritability=0.990` (capped; true h² may be 1) or `0.000`. At 0 the
  LMM reduces to OLS (K plays no role); at 0.99 small errors in h move p-values (up to 0.8 in −log10 p at
  h² = 0.99 in Phase 4).
* **Near-singular K**: few distinct genotypes (inbred strains, replicates) give a K of low rank; pylmm
  floors eigenvalues at 1e-6. Fine for the LMM, but the effective sample size for a SNP is the number of
  distinct genotypes. Check `np.linalg.matrix_rank(K)` and the number of strains carrying the minor allele;
  a hit driven by one or two strains is fragile. Inbred panels with replicates: `estimate-heritability`
  section 7.
* **Small n** (< ~100) or **n close to the number of covariates**: few residual df, unstable h.
* **Low minor-allele count**, high missingness (mean imputation pulls toward 0), SNPs with t far above
  their LD neighbours (genotyping error).
* **ML fits** (`--ml`) of a centred K from more SNPs than individuals: unbounded likelihood, h at 0.99.
* **Different allele coding** between runs or tools: check signs with the `.bim`.

## 6. Loci and comparing with published loci

**From SNPs to loci.** Neighbouring SNPs in LD share a signal; report loci, not SNPs. A simple,
LD-free definition: take the best SNP, make every SNP within a window of it (e.g. ±1 Mb, or the LD
decay distance of the panel) part of its locus, and repeat with the best SNP left. `summarize_gwas.py
--clump-kb KB` does this with the `.bim` positions (`--clump-snps N`: ±N SNPs in file order, when there
are no positions); `--clump-p` starts loci only below a threshold:

```bash
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv --bfile scan --per-allele --clump-kb 100 --clump-p 1e-4
```

With `--plot`, the same command also draws the Manhattan and QQ plots (needs matplotlib, in the `sims`
group):

```bash needs=matplotlib
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv --bfile scan --clump-kb 100 --top 1 --plot scan.png
```

```python
loci = summarize_gwas.clump(res, summarize_gwas.read_bim("scan"), kb=100, p_max=1e-4)
lead = loci[0]
print("%d loci; first: lead SNP %d, %d SNPs" % (len(loci), lead["lead"], len(lead["members"])))
assert lead["lead"] == 10 and 10 in lead["members"]
```

In inbred panels, LD extends over megabases and the window should too; for an LD-based locus (an LD
interval around the lead), compute `r²` with the lead SNP over the window from the genotypes.

**Comparing with published loci.**

* Match by **position on the same genome build** (lift over if needed), not by SNP ID alone; for SNP
  arrays, rsIDs may differ between releases.
* Expect the *region* to replicate rather than the same lead SNP: compare the best p in a window (e.g. ±1
  Mb, or the published LD interval) and the direction of effect of the same allele.
* Differences in −log10 p of a few units are normal across methods (kinship construction, covariates,
  REML vs ML, refit, LOCO vs not, sample QC). A sign flip at the same allele, or no signal anywhere in the
  window, needs explaining.
* Compare h² only after matching the kinship scale and the unit (individual vs strain mean), see
  `estimate-heritability` sections 3 and 7.
