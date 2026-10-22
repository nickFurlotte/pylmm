---
name: interpret-gwas-results
description: Interpret pylmm GWAS output - the summary-statistics table (GWAS-SSF columns - effect_allele, beta per copy of A1, standard_error, effect_allele_frequency, p_value, neg_log_10_p_value, n, t_stat, beta_std), the JSON run record, the legacy SNP_ID/BETA/BETA_SD/F_STAT/P_VALUE columns and their per-SD A2 scale, comparing effects and alleles with plink2, GEMMA, GCTA, REGENIE or BOLT-LMM, genome-wide significance thresholds (Bonferroni, effective number of tests, a parametric-bootstrap threshold for related samples), genomic control lambda_GC and QQ plots, comparing hits with published loci, and when not to trust a result (near-singular kinship, h2 at the 0 or 1 boundary, small n, rare genotypes, inbred panels). Use when reading, summarising, thresholding or comparing pylmm association results.
---

# Interpret pylmm GWAS results

The blocks below run in CI, in order, from the repository root. `scripts/summarize_gwas.py` (this
skill) prints counts, λ_GC, the Bonferroni threshold and the top SNPs of an output file of either format
(ranked by -log10 p, which stays exact where p underflows). Options: `--clump-kb KB` / `--clump-snps N`
group hits into loci (section 6); `--plot out.png` draws a Manhattan plot (x = base-pair position) and a
QQ plot (needs matplotlib). For legacy files only: `--bfile PREFIX` joins chromosome, position and
alleles from the `.bim`, and `--per-allele` converts BETA (section 2).

## 1. The output files

`pylmm gwas -o out.tsv.gz` writes two files: the results table and a JSON run record, `out.json`.

**The table** (gzip-compressed when the name ends in `.gz`; tab-separated, one line per SNP in
genotype-file order). Its first eight columns are the mandatory columns of the GWAS Catalog format
GWAS-SSF v1.1.0, in its order; the rest are optional GWAS-SSF columns and pylmm's own:

| column | meaning | careful |
|---|---|---|
| `chromosome`, `base_pair_location` | from the `.bim` (`.map`; `--emma-map` for EMMA input) | as written in the input (`X`, not 23); `NA` for EMMA input without a map |
| `effect_allele` | **A1**, `.bim` column 5: ALT for plink2-written files; for EMMA input the allele coded 1 | the minor allele for plink 1.9 files without `--keep-allele-order` |
| `other_allele` | A2, `.bim` column 6 | PLINK's `0` (no second allele) is written `NA` |
| `beta` | effect of **one copy of the effect allele** | per allele, like plink2 `--glm` (whose default A1 is the minor allele: `omit-ref` gives ALT) |
| `standard_error` | SE of `beta` | |
| `effect_allele_frequency` | over the tested individuals with a call | not the panel frequency |
| `p_value` | two-sided Wald, from t(n - q - 1) | below 1e-300 written from -log10 p (`3.16228e-401`, 6 digits), never 0 |
| `variant_id` | the input's SNP ID | not GWAS-SSF's `chr_pos_ref_alt` |
| `rsid` | the SNP ID if it is `rs` + digits, else `NA` | |
| `n` | individuals the SNP was tested on | per SNP with `--remove-missing-genotypes` |
| `t_stat` | the test statistic, signed for the effect allele (Wald: `beta / standard_error`) | `--test lrt`: signed √χ² (sign of `beta`), not `beta / standard_error` |
| `neg_log_10_p_value` | -log10 p | rank by this, not by `p_value` |
| `missing_rate` | share of tested individuals without a call | mean-imputed unless `--remove-missing-genotypes` |
| `beta_std` | `beta` × SD of the effect allele count over the called genotypes | the effect of one SD of the genotype |

`NA` is missing: an untestable SNP (constant, collinear with covariates, all missing) has `NA` from `beta`
to `p_value`. `pylmm.sumstats.read_sumstats` (or `pandas.read_csv(f, sep="\t")`) reads it.

**The run record** (`out.json`, schema `pylmm.gwas.run/1`; keys documented in `pylmm/cli/run_record.py`):
`command`, `inputs` (each file's size, modification time and SHA-256), `samples` (genotyped, analysed,
and dropped by reason), `model`, `kinship` (its scale), `null_fit` (h, σ², σ²_g, σ²_e, log-likelihood;
per chromosome in `loco` with `--loco`), `results` (`lambda_gc`, `n_tested`, `not_tested` by reason),
`warnings` (every warning the run logged) and `timings_s`. `status` is `"error"` (with the message) if
the run failed. Read it first: it answers "which individuals, which K, which fit" without the log.

**The legacy table** (`--output-format legacy`, pylmm 1.0.0's): `SNP_ID`, `BETA` (effect of the **A2**
allele **per SD** of the standardised genotype), `BETA_SD` (its standard error, not an SD), `F_STAT` (the
signed Wald t, not an F) and `P_VALUE` (exactly 0.0 when it underflows); `nan` for untestable SNPs.

**The LRT** (`--test lrt`, `gwas(test="lrt")`): `t_stat` (legacy `F_STAT`) is the signed square root
of the likelihood-ratio statistic (the sign of `beta`) and `p_value` (`P_VALUE`) is from chi2(1), also
for `neg_log_10_p_value` where p underflows; `beta` / `standard_error` (`BETA` / `BETA_SD`) come from the
ML fit with the SNP, so `beta / standard_error` is not `t_stat`. The run record's `null_fit` is then the
LRT's reduced ML fit (`kind: lrt-reduced-ml`, `reml: false`, with `dropped_null_directions`), and SNPs
whose ML fit has no maximum are counted in `results.not_tested.lrt_unbounded`. Which to use:

* **Wald, null fit (default)**: the genome-wide scan. Calibrated in the simulations (`sims/studies/s9_tests.md`).
* **Wald with `--refit`**: at strong loci, where the null fit's h absorbed part of the SNP's effect and the
  default p-value is conservative; and to compare with EMMA or GEMMA (`-lmm 1` is a refitted Wald test).
  Slightly anti-conservative at n ~ 100 (1.2-1.4x the nominal rate at 1e-4 / 1e-5 in S9).
* **LRT** (`--test lrt`): ML fits with and without each SNP, chi2(1). With pylmm's centred kinship
  and an intercept the plain ML likelihood has no maximum (it grows as h -> 1), and both fits would sit at
  h = 1 in most replicates whatever the true h², making the LRT anti-conservative (S9: λ_GC up to 1.14,
  2.4x the nominal rate at 1e-5). pylmm's LRT therefore leaves out the kinship's null directions that the
  covariates span (reduced ML): calibrated like the refitted Wald test (λ_GC 0.99-1.03; 1.2-1.35x at 1e-4 /
  1e-5 at n = 100, no correction for the fitted variance component). It differs from GEMMA's `-lmm 2`,
  which centres K and keeps the plain ML (its fits sit at its bound lambda = 1e5). In panels where most
  strains have one animal and a few have two, the LRT can still be ill-posed: a warning that its null
  model has no maximum (then its p-values are anti-conservative), or `nan` lines for SNPs whose calls
  differ within a strain (counted in the log). Use the Wald test for such panels. A `nan` LRT line is a
  "not tested", not a non-significant SNP.
* **Score test**: not a `--test` option; `LMM.score_batch` reproduces GEMMA's `-lmm 3`. It is a monotone
  function of the default Wald statistic (same ranking), conservative in the tail (n = 100: 0.1x the
  nominal rate at 1e-5), and its p-value has a floor (t² <= n - q: about 9.7e-6 at n = 30, 7.4e-9 at n =
  50), so a GEMMA `-lmm 3` scan of a small panel cannot reach genome-wide significance.

A SNP significant under one test and far from it under another is a SNP to look at (a strong effect, a
boundary h, few distinct genotypes).

```python
import json
import sys
import numpy as np
from scipy import linalg, stats
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
sys.path.insert(0, "skills/interpret-gwas-results/scripts")
import plink_tools, summarize_gwas
from pylmm.lmm import LMM, calculate_kinship, gwas
from pylmm.sumstats import read_sumstats

# a small simulated scan: 300 unrelated people, 1000 SNPs on 2 chromosomes (10 kb apart),
# SNP 10 causal (each copy of A2 raises y by 0.5)
rng = np.random.default_rng(1)
G = rng.binomial(2, rng.uniform(0.05, 0.5, 1000), size=(300, 1000)).astype(float)
y = 0.5 * G[:, 10] + rng.normal(size=300)
iids = ["i%d" % i for i in range(300)]
plink_tools.write_bed("scan", G, iids=iids, chrom=np.repeat(["1", "2"], 500),
                      pos=np.tile(np.arange(1, 501) * 10_000, 2),     # G = copies of A2
                      a1=["C"] * 1000, a2=["T"] * 1000)
plink_tools.write_pheno("scan.pheno", [(i, i) for i in iids], y, ["y"])
```

```bash
pylmm kinship --bfile scan -o scan.kin -v
pylmm gwas --bfile scan --kinship scan.kin --pheno scan.pheno -o scan.tsv.gz -v
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv.gz --top 3
```

```python
res = read_sumstats("scan.tsv.gz")
with open("scan.json") as f:
    run = json.load(f)
print("analysed %d, h = %.3f, lambda_GC %.3f" % (
    run["samples"]["n_analysed"], run["null_fit"]["h"], run["results"]["lambda_gc"]))
assert run["results"]["n_variants"] == len(res["beta"]) == 1000
```

## 2. Effect allele and scale

`beta` is per copy of `effect_allele` (A1). The simulation made A2 the causal allele (+0.5 per copy), so
`beta` of SNP 10 is about **-0.5**: the same effect, seen from the other allele.

```python
print("SNP 10: effect allele %s, beta %.3f (SE %.3f), other allele %s" % (
    res["effect_allele"][10], res["beta"][10], res["standard_error"][10], res["other_allele"][10]))
assert abs(res["beta"][10] + 0.5) < 3 * res["standard_error"][10]
eaf = (2 - G).mean(axis=0) / 2                                     # A1 frequency, from the genotypes
assert np.allclose(res["effect_allele_frequency"], eaf)
```

To compare with another study: line up the alleles by name, then flip the sign of `beta` (and use
`1 - effect_allele_frequency`) where the effect alleles are swapped; t and p need no conversion (t flips
sign with the allele). A palindromic SNP (A/T, C/G) cannot be aligned by name alone: use the
frequencies, or drop it.

| tool | effect allele in its output | scale |
|---|---|---|
| pylmm (default table) | `effect_allele` = A1 (`.bim` col 5; ALT for plink2 files) | per allele |
| pylmm `--output-format legacy`, pylmm 1.0.0 | A2 (`.bim` col 6) | per SD |
| plink2 `--glm` | `A1` column: the minor allele by default, ALT with `omit-ref` | per allele |
| GEMMA `-lmm` | `allele1`, usually `.bim` A1 (not verified here: check the GEMMA manual for which allele `beta` refers to) | per allele |
| GCTA `--mlma`, fastGWA | `A1` | per allele |
| REGENIE | `ALLELE1` (ALT) | per allele |
| BOLT-LMM | `ALLELE1` | per allele |
| SAIGE | `Allele2` | per allele |

With an identity kinship (no relatedness), pylmm's `beta`, SE, t, p, frequency and `n` equal plink2 `--glm
omit-ref` to its printed precision (tests/test_sumstats.py). plink 1.9 without `--keep-allele-order`
makes A1 the minor allele, so the effect allele can differ between datasets. Compare alleles by name,
never by position.

**How `beta` is obtained** (and why it is on one scale for every SNP): the model tests the standardised
genotype `x = (g - m) / s` of the A2 fraction `g` (0, 0.5, 1), so its coefficient is per SD; per copy of
A1 (`c = 2 - 2g`) it is `beta_x * dx/dc = -beta_x / (2 s)`, with `s` the SD the genotype was standardised
by (or 1 for a SNP tested raw under `--remove-missing-genotypes`). `beta_std` is `beta` times the SD of the
A1 count, the same definition for every SNP whatever its missing calls.

**Legacy files.** BETA of the legacy table is per SD of the A2 dosage. Per copy of A1: `-BETA / sd`, with
`sd` the SD of the 0/1/2 dosage over the called individuals of the genotype file (`summarize_gwas.py
--bfile PREFIX --per-allele` does this from `PREFIX.bed`; approximate if the kinship dropped individuals,
wrong for `--remove-missing-genotypes`, whose BETA is per 0-1 step of the A2 fraction for SNPs without
missing calls). The default table needs none of this.

```bash
pylmm gwas --bfile scan --kinship scan.kin --pheno scan.pheno -o scan_legacy.tsv --output-format legacy
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan_legacy.tsv --bfile scan --per-allele --top 3
```

```python
leg = summarize_gwas.read_results("scan_legacy.tsv")
sd = summarize_gwas.dosage_sd("scan")                             # SD of the A2 dosage, from the BED
assert np.allclose(-leg["beta"] / sd, res["beta"], rtol=1e-10, equal_nan=True)
assert np.array_equal(-leg["t"], res["t_stat"], equal_nan=True)
```

**Submitting to the GWAS Catalog.** The table follows GWAS-SSF's column order but is not a submittable
file as it is (the deviations are listed in `pylmm/sumstats.py`): untested SNPs are `NA` rows, alleles
and chromosomes are written as in the input (`1`/`2` alleles, `X`, mouse chromosome numbers), and
`variant_id` is the input's ID. For human data on a known build, keep the tested rows with ACGT alleles
on chromosomes 1-25 (X = 23, Y = 24, MT = 25), build `variant_id` as `chr_pos_ref_alt` from your REF/ALT
(here A2/A1, as plink2 writes them), write `#NA` for the remaining missing values, then run the
Catalog's validator (`gwas-ssf validate`, from `gwas-sumstats-tools`) and write its metadata YAML:

```python
import re

ssf = read_sumstats("scan.tsv.gz")
keep = (
    np.isfinite(ssf["p_value"])
    & np.array([bool(re.fullmatch("[ACGT]+", a)) for a in ssf["effect_allele"]])
    & np.array([bool(re.fullmatch("[ACGT]+", a)) for a in ssf["other_allele"]])
)
chrom = np.array([{"X": "23", "Y": "24", "MT": "25"}.get(c, c) for c in ssf["chromosome"]])
keep &= np.isin(chrom, [str(k) for k in range(1, 26)])
with open("scan.ssf.tsv", "w") as f:
    cols = ["chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
            "standard_error", "effect_allele_frequency", "p_value", "variant_id", "rsid", "n"]
    f.write("\t".join(cols) + "\n")
    for j in np.flatnonzero(keep):
        vid = "%s_%d_%s_%s" % (chrom[j], ssf["base_pair_location"][j], ssf["other_allele"][j],
                               ssf["effect_allele"][j])          # chr_pos_ref_alt (REF = A2)
        row = [chrom[j], "%d" % ssf["base_pair_location"][j], ssf["effect_allele"][j],
               ssf["other_allele"][j]] + [repr(float(ssf[c][j])) for c in cols[4:8]]
        rs = ssf["rsid"][j]
        f.write("\t".join(row + [vid, "#NA" if rs == "NA" else rs, "%d" % ssf["n"][j]]) + "\n")
print("%d of %d rows kept" % (keep.sum(), len(keep)))
assert keep.sum() == np.isfinite(ssf["p_value"]).sum()
```

(The p-values below 1e-300 that the table writes from -log10 p read back as 0.0 here: copy those strings
from the table, or give `neg_log_10_p_value` as column 8 instead of `p_value`.)

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
lam_gc = summarize_gwas.lambda_gc(res["p_value"])
print("lambda_GC = %.3f" % lam_gc)
assert 0.8 < lam_gc < 1.25
```

* λ > 1.1 with an LMM: missing covariates (batch, sex), a kinship that does not match the samples (wrong
  order: check the `.id` file), or a K from too few or the wrong SNPs. Look at the QQ plot: a lifted whole
  curve is inflation, a tail-only departure is signal. (The rule that λ grows with n·h²/M under
  polygenicity is the linear-regression, no-LMM result. Under an LMM whose K includes the tested SNPs,
  polygenic signal tends to *deflate* the test statistics instead, because K absorbs it; LOCO is the
  remedy: `pylmm gwas --loco`, or `pylmm.loco` in the library.)
* **A LOCO scan** (`pylmm gwas --loco`, `pylmm.loco`): λ_GC of a polygenic trait is above 1 with no
  confounding at all, because the SNPs linked to the tested chromosome's own causal variants are not
  explained by `K_{-c}`: S8 measured 1.1-1.5 on the HMDP strains and up to 2.8 on a synthetic inbred
  panel, and the README mice give ~1.5 at h² = 0.5, while a null phenotype gives ~1.0. So under LOCO
  λ_GC is not a calibration check (an LD-score-style intercept would be; pylmm has none), and **never
  apply genomic control to a LOCO scan**: it would remove the linked signal LOCO recovers. Check
  calibration with a permuted or simulated null phenotype, or on chromosomes without signal.
* λ < 1: K built from the tested SNPs absorbing signal (proximal contamination; LOCO, `--loco`, helps), or h² = 0 fits
  (mildly conservative in the simulation studies).
* **Few effective tests** (inbred panels, small scans, one chromosome): λ from one scan is noise (simulation studies:
  5–95% range 0.7–1.5 under the null on the bundled mice). Do not GC-correct LMM results by default.

## 5. When to distrust a result

* **h at a boundary**: the run record's `null_fit.h` (log: `null model: heritability=...`) at 1 or 0 (with `--legacy-fit`, as in pylmm 1.0.0,
  the upper end is a cap at `0.990`). At 0 the LMM reduces to OLS (K plays no role); near 1 small errors
  in h move p-values (up to 0.8 in −log10 p at h² = 0.99 in the simulation studies). A log warning "h is
  at its upper boundary" (h ≈ 1, i.e. ≥ 0.999 for a K of the usual scale, with a rank-deficient K, e.g. replicate animals of inbred strains)
  means the test is anti-conservative: λ_GC was 3.6 in a simulation with true h² = 1 on 300 mice of about
  100 strains. Test strain means instead.
* **Near-singular K**: few distinct genotypes (inbred strains, replicates) give a K of low rank; pylmm
  floors eigenvalues at 1e-6. Fine for the LMM, but the effective sample size for a SNP is the number of
  distinct genotypes. Check `np.linalg.matrix_rank(K)` and the number of strains carrying the minor allele;
  a hit driven by one or two strains is fragile. Inbred panels with replicates: `estimate-heritability`
  section 7.
* **Small n** (< ~100) or **n close to the number of covariates**: few residual df, unstable h.
* **Low minor-allele count**, high missingness (mean imputation pulls toward 0), SNPs with t far above
  their LD neighbours (genotyping error).
* **ML fits** (`--ml`) of a centred K from more SNPs than individuals: unbounded likelihood (pylmm warns),
  h near 1 or an arbitrary interior local maximum.
* **Different allele coding** between runs or tools: check `effect_allele` before comparing signs.

## 6. Loci and comparing with published loci

**From SNPs to loci.** Neighbouring SNPs in LD share a signal; report loci, not SNPs. A simple,
LD-free definition: take the best SNP, make every SNP within a window of it (e.g. ±1 Mb, or the LD
decay distance of the panel) part of its locus, and repeat with the best SNP left. `summarize_gwas.py
--clump-kb KB` does this with the table's positions (`--clump-snps N`: ±N SNPs in file order, when there
are no positions); `--clump-p` starts loci only below a threshold:

```bash
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv.gz --clump-kb 100 --clump-p 1e-4
```

With `--plot`, the same command also draws the Manhattan and QQ plots (needs matplotlib, in the `sims`
group):

```bash needs=matplotlib
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py scan.tsv.gz --clump-kb 100 --top 1 --plot scan.png
```

```python
table = summarize_gwas.read_results("scan.tsv.gz")
loci = summarize_gwas.clump(table, summarize_gwas.positions(table), kb=100, p_max=1e-4)
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
* For a written comparison, fill in [comparison-worksheet.md](comparison-worksheet.md) first: kinship
  formula and scale, observation unit, transform, null-only vs per-SNP refit, build, named allele, SNP
  set, thresholds and the pre-registered rule for "replicated". In the [replication](../../docs/replication.md) of two EMMA-era
  mouse GWAS, an IBS-style K with a per-SNP refit came closest to the published p-values.
