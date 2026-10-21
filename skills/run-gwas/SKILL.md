---
name: run-gwas
description: Run a genome-wide association scan (GWAS) with pylmm's linear mixed model, end to end - from PLINK BED/TPED or EMMA genotypes, a phenotype file and a kinship matrix to a checked result table - with the pylmm gwas command line or the Python library. Covers choosing REML vs ML, --refit, covariates and the intercept, sample ID matching, phenotype and SNP QC recipes, first sanity checks of the output (interpretation, thresholds and effect alleles are in interpret-gwas-results), inbred panels, and fixing the common pylmm errors. Use when asked to run, rerun, script or debug a GWAS or association scan with pylmm.
---

# Run a GWAS with pylmm

The steps: genotypes in PLINK BED → phenotype file → kinship (`pylmm kinship`, writes `K.id`) → scan
(`pylmm gwas`) → check. Paths in the examples are relative to the repository root, and the blocks run in
CI in order (`tests/test_docs.py`). Background: [docs/guide.md](../../docs/guide.md). Recipes for phenotype
QC and transformation, SNP QC (MAF, missingness) and defining loci: [reference.md](reference.md).

Run the commands with `-v` while setting up an analysis: it logs how many individuals were read,
matched and dropped (missing phenotype or covariate, not in the kinship), alignment warnings and the null
fit. `-q` hides warnings that matter for correctness (e.g. a kinship without `.id` assumed to be in
genotype order); use it only for a pipeline you have already checked.

## 0. Inputs

| input | format | notes |
|---|---|---|
| genotypes | PLINK 1 BED/BIM/FAM (`--bfile`), TPED (`--tfile`) or EMMA (`--emma-geno`) | anything else: the `convert-genotype-formats` skill |
| phenotype | `FID IID name1 name2 ...` (header optional, `NA`/`-9` missing) | matched to the `.fam` by FID **and** IID |
| covariates | same layout; only `NA` is missing | an intercept is added for you |
| kinship | n × n text from `pylmm kinship`, plus `K.id` | `K.id` aligns K by ID (PLINK input); with `--emma-geno` K is in genotype order, no `.id` and no warning |

Make a small example: 3000 SNPs of the bundled mice, and a trait with a planted effect on SNP 500.

```python
import itertools, subprocess, sys
import numpy as np
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools
from pylmm.plink_io import PlinkReader

SRC = "data/snps.132k.clean.noX"
plink_tools.subset_bed(SRC, "geno", np.arange(3000))
with PlinkReader("geno", pheno_file=SRC + ".fake.phenos") as r:
    x = next(itertools.islice(r.iter_snps(), 500, None))[0]     # standardised SNP 500
    y = r.phenos[:, 0] + 0.8 * x
    ids = r.indivs
plink_tools.write_pheno("geno.pheno", ids, y, ["trait"])
```

## 1. Kinship

```bash
pylmm kinship --bfile geno -o geno.kin -v      # writes geno.kin and geno.kin.id
```

Build K from the SNPs you trust (QC'd, autosomal). With many SNPs, K from all of them is standard; to
leave the tested chromosome out of K (LOCO) use `pylmm gwas --loco` instead of a kinship file (section
2). The default
`--kinship-type standardized` is the GRM (GEMMA `-gk 2`'s formula on complete genotypes; GEMMA
standardises after mean imputation and drops MAF < 0.01 / missingness > 0.05 SNPs by default);
`--kinship-type ibs` is EMMA's IBS kinship (to match an EMMA-era inbred-panel paper); `centered` is GEMMA
`-gk 1`'s formula (0/1/2 counts). `-v` logs the scale of K (`kinship scale: mean diagonal
...`): the fitted h is on that scale (`estimate-heritability` section 3). The library's
`calculate_kinship` gives the same K from the raw genotypes. A K from another tool: a GCTA GRM
(`gcta --make-grm`, `plink2 --make-grm-bin`) is read natively with `--grm PREFIX` in place of
`--kinship`, a plink2 `--make-rel` file with `--kinship X.rel` (or `.rel.bin`), both aligned by their
IDs; a plain n × n text matrix (GEMMA) needs its `.id` file (`pylmm.samples.write_id_file`); see
`convert-genotype-formats`. `pylmm kinship --out-format grm -o geno` writes a GRM (float32: scans on it
match the text K's to about 1e-7).

**Inbred panels** (strains with replicate animals, few distinct genotypes): a K built from the tested
SNPs absorbs much of a causal SNP's signal, so power is low, and a polygenic trait deflates the whole
scan on chromosomes without causal variants (S8, synthetic inbred panel: λ_GC 0.57-0.66 at h² 0.6-0.9
with K from all SNPs, 1.03 with LOCO). **For inbred panels, use `pylmm gwas --loco`** (section 2): power
at a q² = 0.05 QTL roughly doubled in S8. Expect its λ_GC to be well above 1 for a polygenic trait, which
is linked signal, not inflation (section 3); never genomic-control it. `simulate-lmm-data` section 6 shows the size of the effect and the disjoint-K check, and
`estimate-heritability` section 7 the individual vs strain-mean set-up. Preparing such data (strain
aliases, expanding strains to animals, IDs with spaces, imputing over strains rather than animals): the
`inbred-strain-panels` skill.

## 2. Scan

```bash
pylmm gwas --bfile geno --kinship geno.kin --pheno geno.pheno --pheno-name trait -o trait.tsv -v 2> trait.log
grep "null model" trait.log
```

Covariates come from a file with the same layout; with a `FID IID name ...` header, pick columns by name
(here a made-up `sex` and `batch`; only `sex` is used):

```python
rng = np.random.default_rng(0)
cov = np.column_stack([rng.integers(0, 2, len(ids)), rng.normal(size=len(ids))])
plink_tools.write_pheno("geno.covar", ids, cov, ["sex", "batch"])
```

```bash
pylmm gwas --bfile geno --kinship geno.kin --pheno geno.pheno --pheno-name trait --covar geno.covar --covar-name sex -o trait_sex.tsv -v 2> trait_sex.log
grep -i "covariate\|individuals\|null model" trait_sex.log
```

Options that matter:

| option | default | when to change it |
|---|---|---|
| `--reml` / `--ml` | REML | ML only to match a method that used ML; ML is ill-posed for a centred K from more SNPs than individuals |
| `--refit` | off (fit h once, EMMAX/P3D) | strong-effect loci (the null fit absorbs part of a big SNP's effect, so P3D is conservative there), or to match EMMA / GEMMA (`-lmm 1`); the exact per-SNP fit, a block at a time (12-24x the cost of the default scan; `--no-batch` or `--legacy-fit`: the old grid search per SNP, 11-61x slower still). A little more power than the default (S9), mildly anti-conservative at n ~ 100 (1.2-1.4x at 1e-4 / 1e-5). (pylmm ≤ 1.0.0's per-SNP fits crashed for 0.25–2.8% of SNPs at h² 0.8–0.99, [phase4](../../docs/phase4-simulations.md); fixed since) |
| `--test wald\|lrt` | `wald` | `lrt`: likelihood ratio of ML fits with and without the SNP, chi2(1), always ML (`--reml`/`--ml`/`--refit` do not apply), with the kinship's null directions that the covariates span left out so that ML has a maximum (pylmm's centred K and an intercept: one direction). Calibrated and powered like `--refit` ([S9](../../sims/studies/s9_tests.md)); 7.5-15x the default scan's cost. `F_STAT` is then the signed root of the statistic (`interpret-gwas-results`). The score test is the library's `LMM.score_batch` (GEMMA `-lmm 3` comparisons only) |
| `--covar F --covar-name a b` | none | covariates; the intercept is added unless `--no-intercept` |
| `--pheno-name N` / `--pheno-col N` | first column | `--pheno-col` is 1-based |
| `--id-match iid` | `fid-iid` | the phenotype/covariate/kinship FIDs differ from the `.fam` |
| `--remove-missing-genotypes` | mean-impute | test each SNP on its called individuals (slower; changes BETA's scale) |
| `--loco` | off (`--kinship` K for every SNP) | test each chromosome's SNPs with a K built from the other chromosomes (below); **recommended for inbred panels**; costs one eigendecomposition per chromosome |
| `--eigen eig.npz` | recompute | many phenotypes on the same individuals (`pylmm kinship --eigen-npz`; the file carries the sample IDs, which must be the tested individuals: an error otherwise) |
| `--block-size N` | 1024 | SNPs read (a BED file is decoded a block at a time) and tested together (one matrix product per block); lower it if memory is tight (about 6-8 × n × N floats, with `--refit` and `--test lrt` too) |
| `--no-batch` | batched | one SNP at a time, the pylmm 1.0.0 code path (slower; with `--refit`, the grid search per SNP); SNPs with missing calls always run this way. With well-conditioned covariates both give the same numbers up to rounding (within ~1e-9 relative); with a badly scaled covariate (a raw date or year column) they can differ by up to a few 1e-6 relative, and the batched default is the more accurate. Centre or scale such covariates (`(c - c.mean()) / c.std()`): the model and its tests are the same, and both paths are then accurate |

**Leave one chromosome out.** `--loco` replaces `--kinship`: pylmm reads the genotypes once to build
`K_{-c}` (the kinship of every chromosome but `c`, `--kinship-type` as in `pylmm kinship`) for each
chromosome, then scans, fitting the null model once per chromosome (`-v` logs each chromosome's K and h).
It needs the chromosome of each SNP (the `.bim`; `--emma-map FILE` for EMMA input) with each chromosome's
SNPs together, and at least two chromosomes. The example `geno` has chromosome 1 only, so make a file
set with SNPs from three chromosomes:

```python
with PlinkReader(SRC) as r:
    chrom = r.variants.chrom
three = np.flatnonzero(np.isin(chrom, ["1", "2", "3"]))[::10]    # every 10th SNP of chromosomes 1-3
plink_tools.subset_bed(SRC, "geno3", three)
```

```bash
pylmm gwas --bfile geno3 --loco --pheno geno.pheno --pheno-name trait -o trait_loco.tsv -v 2> trait_loco.log
grep "chromosome\|null model" trait_loco.log
```

Cost: one eigendecomposition and null fit per chromosome. That is about the standard run at n ≈ 1200
(the README mice, 19 chromosomes) but 7 times it at n = 3000 (10 chromosomes), and it grows as n³ ×
chromosomes. Memory is about (chromosomes + 6) × n² × 8 bytes, logged before the kinship pass. More than
`--loco-max-chroms` (default 100) chromosomes is an error (usually contigs, or a misread map), and labels
such as `chr1`, `01` and `1` are one chromosome. The genotypes are read in full for the kinships first,
so a genotype read error stops the run before any output. Per-chromosome kinship files for `--loco` are
not supported: with your own kinships, loop over chromosomes in Python (`custom-association-analysis`).
LOCO is opt-in: without `--kinship`, `--grm` or `--loco` the command stops with a usage error that names
all three.

The same scan in Python, when you need the arrays (subsets, loops, other covariates):

```python
from pylmm.lmm import gwas

with PlinkReader("geno", k_file="geno.kin", pheno_file="geno.pheno") as r:   # K aligned by geno.kin.id
    W = np.column_stack([g for g, _ in r.iter_snps()])
    Y, K = r.phenos[:, 0], r.K
ts, ps = gwas(Y, W, K)                     # X0 defaults to the intercept; REML
out = np.genfromtxt("trait.tsv", names=True, dtype=None, encoding="utf-8")
assert np.allclose(ps, out["P_VALUE"], rtol=1e-6, equal_nan=True)   # the CLI and the library agree
```

With covariates in Python, prepend the intercept yourself:
`X0 = np.column_stack([np.ones(len(Y)), r.read_covariates("covar.txt")])` (drop rows with a missing
covariate first). Without it the model has no intercept.

## 3. Check the result

```bash
uv run python skills/interpret-gwas-results/scripts/summarize_gwas.py trait.tsv --bfile geno --top 5
```

```python
sys.path.insert(0, "skills/interpret-gwas-results/scripts")
import summarize_gwas

s = summarize_gwas.summarize(summarize_gwas.read_results("trait.tsv"))
print("lambda_GC %.2f, min p %.1e, top %s" % (s["lambda_gc"], s["min_p"], s["top"][0]["snp"]))
assert s["top"][0]["p"] < 1e-10
```

Checklist:

1. **Sample counts.** `-v` logs how many individuals were read and dropped (missing phenotype or
   covariate, not in the kinship). Unexpected drops usually mean an ID mismatch.
2. **The null fit** line (`null model: heritability=..., sigma=...`): h at 0 or 1 is a boundary estimate
   (see `estimate-heritability`).
3. **λ_GC** near 1 for a polygenic trait. Inflation (> 1.1) with an LMM suggests a missing covariate or a
   K that does not match the samples. (λ growing with n·h²/M is the linear-regression, no-LMM result; with
   an LMM whose K includes the tested SNPs, polygenic signal tends to deflate λ, and LOCO (`--loco`) is
   the remedy.) **Under `--loco`, λ_GC of a polygenic trait is above 1 by design** (linked signal on the
   tested chromosome; ~1.5 on the README mice at h² = 0.5): not a calibration check, and never apply
   genomic control to it (`interpret-gwas-results`).
   In inbred panels
   with few effective tests, λ from one scan is noisy (simulation studies: 5–95% range 0.7–1.5 under the null). The
   example above prints λ ≈ 0.4: 3000 SNPs of one chromosome of an inbred panel are a handful of
   independent tests, and the planted SNP is also in K, which absorbs part of its signal (proximal
   contamination). Such a λ says nothing.
4. **QQ plot** (`--plot out.png`): deviation only in the tail is signal; a shifted whole curve is
   inflation.
5. **Top hits**: are they clusters of SNPs in LD (expected) or isolated singletons with rare genotypes
   (suspect: check the minor allele count)? `nan` rows are untestable SNPs (constant, collinear, all
   missing).
6. Thresholds and effect directions: the `interpret-gwas-results` skill.

## 4. Errors and fixes

Errors are one line, `pylmm: error: ...`, exit code 2:

```bash
pylmm gwas --bfile geno --kinship geno.kin --pheno geno.pheno --pheno-col 2 -o x.tsv 2> err.txt || echo "exit code $?"
cat err.txt
```

| message (abridged) | cause | fix |
|---|---|---|
| `none of the N individuals of the genotype file appear in the phenotype file` | FIDs differ (`0`, or FID = IID from `--double-id`) | `--id-match iid`, or rewrite the FIDs |
| `phenotype file X, line 1: a value is not a number` | a `#FID IID ...` header, or text values | write the header as `FID IID ...`; numeric values only |
| `--pheno-name y: the phenotype file has no FID IID name ... header` | headerless file | add a header or use `--pheno-col` |
| `--pheno-col 2: the phenotype file has 1 phenotype column(s)` | 1-based numbering (old `-p 1` = `--pheno-col 2`) | count from 1 |
| `the covariates (column 0 is the intercept pylmm gwas adds ...): columns 0 and 1 are identical` | the covariate file has its own column of ones | drop it, or `--no-intercept` |
| `... is rank deficient: column j is a linear combination ...` | collinear covariates (e.g. all PCs plus their sum, sex coded twice) | remove the column |
| `kinship file K: expected a n x n matrix ... found ...` | K for other individuals, or a header line | recompute K, or add `K.id` so it is aligned by ID |
| warning `no kinship ID file K.id; assuming the rows ... are the individuals of the genotype file` | K from elsewhere | write `K.id`; never rely on order silently |
| `no phenotype file PREFIX.phenos` | no `--pheno` | pass `--pheno` |
| `... is not in SNP-major order` | old PLINK file | `plink --bfile X --make-bed --out Y` |
| `n=... observations, but the model has q covariate column(s) plus the SNP` | too few individuals | fewer covariates |

Library equivalents raise `ValueError` (file problems: `pylmm.samples.InputError`, a subclass) with the
same text. `LMM.association` on a SNP with `nan` raises `ValueError: array must not contain infs or NaNs`:
use `gwas`, or impute.
