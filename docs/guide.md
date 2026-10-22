# pylmm user guide: ad-hoc analysis with the library

This guide shows how to use pylmm from Python for analyses the `pylmm` command line does not cover:
subsets, custom loops, your own covariates, heritability, and checking results. The command line is
covered too (section 9), because most analyses mix both.

Every code block in this file runs in CI (`tests/test_docs.py`): the blocks of this file run in order, in
one Python namespace, from a scratch directory where `data/`, `docs/`, `sims/` and `skills/` are links to
the repository's folders. So the paths below are relative to the repository root, and a block can use the
variables of the blocks above it. Blocks marked `skip-ci` need something CI does not have (a network
install, plink2).

Contents: 1 install · 2 the model · 3 loading data · 4 kinship · 5 the null fit · 6 one SNP ·
7 scans · 8 missing data · 9 the command line · 10 reading the results · 11 performance ·
12 pitfalls · 13 validated behaviour

## 1. Install

pylmm needs Python 3.11+, numpy and scipy. Install the `main` branch (the older `master` branch keeps the
original Python 2 code):

```bash skip-ci: network install
pip install "git+https://github.com/nickFurlotte/pylmm@main"
# or, in a uv project
uv add "pylmm @ git+https://github.com/nickFurlotte/pylmm@main"
```

Reading PLINK 2 PGEN files (section 3) needs the optional `pgen` extra, `pylmm[pgen]`, which installs
plink2's reader pgenlib (LGPL-3.0; pylmm itself is MIT); from a clone, `uv sync --extra pgen`.

A git install clones the whole repository, including ~215 MB of example data, so it takes a while (what
the data is, and its checksums: [data/README.md](../data/README.md)). From a clone, `uv sync` (add
`--group sims` for the simulation framework) and prefix commands with `uv run`, which uses the `.venv/`
at the repository root (creating it if needed; `UV_PROJECT_ENVIRONMENT` points uv elsewhere):

```bash
uv run pylmm --version
uv run python -c "import pylmm, pylmm.lmm; print(pylmm.__version__)"
```

## 2. The model

For `n` individuals with phenotype `Y`, covariates `X0` (n × q) and kinship `K` (n × n):

```text
Y = X0 b + x a + u + e,    u ~ N(0, h s2 K),    e ~ N(0, (1 - h) s2 I)
```

* `h` is the heritability (the share of the total variance `s2` carried by `K`; see section 5 for when
  that equals the h² of a paper), `s2` is called `sigma` in the code.
* `x` is one SNP. `association` and `gwas` test `a = 0` with a Wald t test, `t = a_hat / se`, against a
  t distribution with `n - q - 1` degrees of freedom.
* By default `h` and `s2` are fitted once without a SNP (the "null fit") and reused for every SNP (EMMAX /
  P3D). `refit=True` (`--refit`) refits them per SNP.
* REML is the default everywhere (`reml=True`, `--reml`); `reml=False` / `--ml` is maximum likelihood.

The modules: `pylmm.lmm` (`LMM`, `gwas`), `pylmm.kinship` (`calculate_kinship`, `kinship_scale`), `pylmm.plink_io` (`PlinkReader`),
`pylmm.genotypes` (variant metadata, `ArraySource`, the `GenotypeSource` interface), `pylmm.samples`
(sample IDs and alignment), `pylmm.misc` (`fit_two`, two kinships). Their docstrings are
complete (numpydoc, with examples): `help(pylmm.lmm.LMM)`.

## 3. Loading data

### PLINK BED/BIM/FAM

`PlinkReader` reads one SNP at a time. Take the first `m` SNPs with `itertools.islice`; reading all 98,740
SNPs of the example as float64 needs about 1 GB.

```python
import itertools
import numpy as np
from pylmm.plink_io import PlinkReader

BED = "data/snps.132k.clean.noX"          # .bed/.bim/.fam: 1219 mice, 98,740 SNPs
with PlinkReader(BED, pheno_file=BED + ".fake.phenos") as reader:
    snps = [(g, name) for g, name in itertools.islice(reader.iter_snps(), 2000)]
    ids = reader.indivs                   # [(FID, IID), ...] in .fam order
    Y_fake = reader.phenos[:, 0]          # phenos is (n, n_phenotypes)

W = np.column_stack([g for g, _ in snps])  # (n, m) genotype matrix
snp_names = [name for _, name in snps]
n = len(ids)
print(W.shape, ids[:2], snp_names[0])
assert W.shape == (1219, 2000) and ids[0] == ("1", "1")
```

**Genotype coding.** By default (`norm_genotype=True`) each SNP is standardised: centred and scaled with the
mean and SD of its called genotypes, then a missing call is set to 0 (the mean). With `norm_genotype=False` you get the raw
value, the **fraction of copies of A2**, the second allele of the `.bim` line (column 6): 0.0, 0.5, 1.0, or
nan for a missing call. It is not a 0/1/2 count. A plink2-written `.bim` has A1 = ALT and A2 = REF, so the
counted allele is REF.

```python
with PlinkReader(BED, norm_genotype=False) as reader:
    g_raw, name = next(reader.iter_snps())
print(name, np.unique(g_raw[~np.isnan(g_raw)]), int(np.isnan(g_raw).sum()), "missing")
assert set(np.unique(g_raw[~np.isnan(g_raw)])) <= {0.0, 0.5, 1.0}
```

**Variant metadata.** `reader.variants` holds the `.bim` columns of every SNP, in iteration order, as
read-only numpy arrays: `chrom`, `snp`, `cm`, `bp`, `a1`, `a2` (a `pylmm.genotypes.Variants`; read once, on
first use; a cM of `NA` is nan, a position that is not an integer is -1, with a logged warning).
`variants[k]` is one SNP as a `Variant` named tuple; a boolean mask selects SNPs, e.g. a chromosome.
For a TPED file the columns come from its `.bim`, or from its `.map` (no alleles); EMMA input has SNP
names only (`SNP_1`, ...), the other columns are `None`.

```python
with PlinkReader(BED) as reader:
    variants = reader.variants
print(len(variants), variants[0])
assert variants.snp[:2000].tolist() == snp_names
on_chr2 = variants.chrom == "2"            # boolean mask over all SNPs
print(int(on_chr2.sum()), "SNPs on chromosome 2, from bp", variants.bp[on_chr2].min())
```

**Blocks of SNPs.** `reader.iter_blocks(block_size)` yields `(G, variants)`: `G` is an `(n, b)` float64
matrix of `b <= block_size` SNPs (the same values as `iter_snps`, bit for bit, standardised or not) and
`variants` is the metadata of those SNPs. A BED file is decoded a block at a time, 2-6x faster than
one SNP at a time (about 2x when standardising, which is done for the whole block at once). A block is
`8 * n * block_size` bytes (default 1024 SNPs: 10 MB for 1219 mice); pick a smaller block for a very
large `n`. Process a genome block by block instead of building the full matrix:

```python
with PlinkReader(BED) as reader:
    G, v = next(reader.iter_blocks(500))
    assert np.array_equal(G, W[:, :500])     # the first 500 SNPs of W above
    sq = {}                                  # sum of squared standardised values per chromosome
    for G, v in itertools.islice(reader.iter_blocks(4096), 3):
        for c in np.unique(v.chrom):
            sq[c] = sq.get(c, 0.0) + float((G[:, v.chrom == c] ** 2).sum())
print(G.shape, v.snp[0], sorted(sq))
```

`skills/convert-genotype-formats/scripts/plink_tools.py` has BED writing and subsetting helpers that pylmm
itself does not have (and `read_bim`, which returns the same columns as a dict):

```python
import sys
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools

bim = plink_tools.read_bim(BED)            # chrom, snp, cm, pos, a1, a2 arrays
assert np.array_equal(bim["pos"], variants.bp) and np.array_equal(bim["a2"], variants.a2)
```

### Other genotype formats

* `PlinkReader(prefix, type="t")`: PLINK TPED/TFAM, alleles coded `1`/`2` (`0` missing; make one with
  `plink --bfile X --recode 12 transpose --out Y`).
* `PlinkReader(path, type="emma")`: an EMMA matrix, one SNP per row, one individual per column, no IDs
  (individuals are positions `0..n-1`, and every other file is matched by position) and no SNP names or
  positions. For the bundled `data/hmdp.liver.snps`, `data/hmdp.liver.snps.map.tsv` gives each row's
  chromosome, build-37 position and alleles (see `data/README.md`).
* PLINK 2 PGEN (`.pgen` + `.pvar` or `.pvar.zst` + `.psam`), with the `pgen` extra:
  `pylmm.pgen.PgenSource(prefix)` (a `GenotypeSource`, with `subset`) or `PlinkReader(prefix,
  type="pgen")` (with the phenotype, covariate and kinship reading of `PlinkReader`; `pylmm gwas --pfile`
  uses it). The genotype is the fraction of copies of **REF**, the A2 of the `.bim` plink2 writes for the
  file: a PGEN reads exactly as `plink2 --pfile X --make-bed` of it (and a BED converted to PGEN reads as
  the BED), and `variants.a2` is REF, `a1` ALT. Imputed **dosages** are read where the file stores them
  (`1 - ALT dosage / 2`, at plink2's 1/16384 precision); `dosage=False` reads the hard calls. A
  multiallelic variant is an `InputError` (drop or split them with plink2 or bcftools first). Without an
  FID column in the `.psam`, the FID is `0`, as in the `.fam` plink2 writes.
* VCF, BCF, BGEN, VCF Zarr, Hail: convert to BED (hard calls) or, to keep dosages, to PGEN; see the
  `convert-genotype-formats` skill.
* Any genotype matrix you already have in numpy works directly with `calculate_kinship`, `LMM` and `gwas`.
  To pass it to code written for blocks (`iter_blocks`), wrap it in `pylmm.genotypes.ArraySource(G,
  sample_ids=..., variants=..., norm_genotype=True)`: the same interface as `PlinkReader`
  (`sample_ids`, `num_samples`, `variants`, `iter_blocks`; the `GenotypeSource` protocol), with the same
  per-SNP standardisation.

```python
with PlinkReader("data/hmdp.liver.snps", type="emma") as emma:
    G_emma = np.column_stack([g for g, _ in itertools.islice(emma.iter_snps(), 100)])
print(G_emma.shape, emma.indivs[:3])       # (94, 100) [0, 1, 2]

from pylmm.genotypes import ArraySource, Variants

counts = np.random.default_rng(1).binomial(2, 0.3, size=(50, 8)).astype(float)   # e.g. from a VCF
source = ArraySource(counts, sample_ids=[("F%d" % i, "I%d" % i) for i in range(50)],
                     variants=Variants(snp=["v%d" % j for j in range(8)], chrom=["1"] * 4 + ["2"] * 4,
                                       bp=np.arange(8) * 1000))
for G, v in source.iter_blocks(5):         # standardised, like PlinkReader's default
    print(G.shape, v.snp.tolist())
```

A PGEN file set (here written by `pylmm.pgen.write_pgen` from the first 500 SNPs of the bundled BED: the
file `plink2 --bfile ... --make-pgen` would make) reads as the BED it came from:

```python needs=pgenlib
from pylmm.pgen import PgenSource, write_pgen

with PlinkReader("data/snps.132k.clean.noX", norm_genotype=False) as bed:
    G500, v500 = next(bed.iter_blocks(500))
    write_pgen("first500", G500, bed.indivs, v500)       # first500.pgen / .pvar / .psam

pg = PgenSource("first500")                              # standardised blocks, like PlinkReader
G_pgen, v_pgen = next(pg.iter_blocks(500))
with PlinkReader("data/snps.132k.clean.noX") as bed:
    G_bed = next(bed.iter_blocks(500))[0]
assert np.array_equal(G_pgen, G_bed) and pg.sample_ids == bed.indivs
print(v_pgen[0])                                         # a2 is REF, the counted allele
```

### Phenotypes and covariates

Phenotype and covariate files are whitespace-separated `FID IID value ...` tables, optionally with a
header line whose first two fields are `FID IID` (any case; `#FID` is not recognised), naming the
columns. Missing values: `NA` and `-9` in phenotypes, only `NA` in covariates.

```python
from pylmm import samples

# a phenotype file with a header, rows in a different order than the .fam, one individual missing
rng = np.random.default_rng(7)
with open("pheno.txt", "w") as f:
    f.write("FID IID weight glucose\n")
    for fid, iid in ids[::-1][:-1]:        # reversed, and the first .fam individual left out
        f.write("%s %s %.3f %.3f\n" % (fid, iid, rng.normal(30, 3), rng.normal(150, 20)))
with open("covar.txt", "w") as f:
    f.write("FID IID sex age\n")
    for i, (fid, iid) in enumerate(ids):
        f.write("%s %s %d %s\n" % (fid, iid, i % 2, "NA" if i == 5 else 8 + i % 5))

import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    reader = PlinkReader(BED, pheno_file="pheno.txt")
    glucose = reader.read_phenos("pheno.txt", columns=["glucose"])[:, 0]
    cov = reader.read_covariates("covar.txt", columns=["sex", "age"])
print(reader.pheno_names, reader.covariate_names, cov.shape)
print(caught[0].message)                   # 1 of 1219 individuals are not in the phenotype file ...
assert np.isnan(glucose[0]) and np.isnan(cov[5, 1])
```

Rows are matched to the genotype file by `(FID, IID)`, never by row position: the file's order does not
matter, its extra individuals are ignored, and individuals it lacks get `nan` (with a warning).
`PlinkReader(..., id_match="iid")` matches on the IID alone, for files whose FIDs disagree (one file uses
`0` as FID, another repeats the IID). Duplicate IDs are an error. `pylmm.samples.align_samples` does the
same alignment for any lists of IDs, for example when you load data with other tools:

```python
al = samples.align_samples({"genotypes": ids, "phenotype": samples.read_sample_table("pheno.txt").ids})
print(al.n, "shared;", al.report()[0])
assert al.n == n - 1
```

`reader.subset(ids)` keeps, and reorders, individuals in `indivs`, `phenos`, `K` and every later genotype
vector together.

## 4. Kinship

`calculate_kinship(W)` (in `pylmm.kinship`, also importable from `pylmm.lmm`) and `pylmm kinship` (the
command line, which also writes `K.id`, the `FID IID` of each row) compute the same kinship from the same
SNPs, to the last bit at the same block size. Three kinds (`kind=` / `--kinship-type`):

| kind | K | use |
|---|---|---|
| `standardized` (default) | `Z Z' / m`, `Z` the SNPs standardised to mean 0 and variance 1 (a GRM; GEMMA `-gk 2`'s formula on complete genotypes) | most analyses |
| `centered` | `C C' / m`, `C` the centred SNPs, not scaled (GEMMA `-gk 1`'s formula on 0/1/2 counts); the CLI counts 0/1/2 copies of A2 (an `--emma-geno` file must be coded 0/0.5/1) | matching GEMMA's default |
| `ibs` | EMMA's identity-by-state kinship: the share of alleles two individuals have in common, diagonal 1 (Kang et al. 2008) | matching EMMA-era inbred-panel papers |

One convention for missing calls: each SNP is centred (and, for `standardized`, scaled) with the mean and
SD of its *called* genotypes, and a missing call is set to the mean, as `PlinkReader`'s standardisation
does; `ibs` follows EMMA (missing calls at the SNP's mean). SNPs with no called genotype, and for
`standardized` and `centered` constant SNPs, are skipped and not counted in `m`. Give
`calculate_kinship` the raw genotypes (`norm_genotype=False`, or 0/1/2 counts): a matrix already
standardised by the reader has its missing calls at 0 and is standardised again.

GEMMA is close but not identical: it mean-imputes a missing call and then (for `-gk 2`) scales by the SD
over all individuals *after* imputation, which is pylmm 1.0.0's library convention (mean diagonal 1 with
missing calls), and by default it first drops SNPs with MAF < 0.01 or missingness > 0.05. So
`standardized` equals GEMMA's `-gk 2` only on complete genotypes passing those filters; `centered`
equals `-gk 1` (0/1/2 counts) on SNPs passing them.

```python
from pylmm.kinship import calculate_kinship, kinship_scale

with PlinkReader(BED, norm_genotype=False) as reader:      # raw 0 / 0.5 / 1 / nan
    W_raw = np.column_stack([g for g, _ in itertools.islice(reader.iter_snps(), 2000)])
K_2000 = calculate_kinship(W_raw)          # from the 2000 SNPs above (67 missing calls in SNP 1 alone)
print(K_2000.shape, kinship_scale(K_2000))
```

To try the command line on the same 2000 SNPs, copy them to a small BED file set (byte for byte, with
`plink_tools.subset_bed`; `plink2 --bfile X --extract snps.txt --make-bed` does the same):

```python
plink_tools.subset_bed(BED, "mini", np.arange(2000))
```

```bash
uv run pylmm kinship --bfile mini -o mini.kin -v 2>&1 | grep "kinship scale"
head -n 2 mini.kin.id
```

```python
K_cli = np.loadtxt("mini.kin")
assert np.array_equal(K_cli, K_2000)                              # the library's K, bit for bit
assert np.allclose(K_cli, W @ W.T / W.shape[1], atol=1e-12)      # = the reader's standardised SNPs
assert samples.read_id_file("mini.kin.id") == ids

# The rest of this guide uses the kinship of all 98,740 SNPs, shipped with the example data
# (`pylmm kinship --bfile data/snps.132k.clean.noX -o full.kin` recomputes it, in ~10 s).
K_FILE = BED + ".pylmm.kin"                 # it has its ID file, K_FILE + ".id"
K = np.loadtxt(K_FILE)
```

**The scale of K** is the scale of `h` (section 5): `kinship_scale(K)` returns its mean diagonal and its
centred mean diagonal `c = trace(P K P) / (n - 1)`, and `pylmm kinship -v` logs both. The standardized K
has a mean diagonal of about 1 - (the missing rate) (0.987 for the bundled BED data, 1.3% missing); the
IBS K has a diagonal of 1 but, for related strains, `c` well below 1 (0.34 on the bundled HMDP strains),
so its `h` is larger for the same data: compare the PVE `h c / (h c + 1 - h)`, not `h`. Nothing is
rescaled for you.

```python
with PlinkReader("data/hmdp.liver.snps", type="emma", norm_genotype=False) as emma:
    G_hmdp = np.column_stack([g for g, _ in itertools.islice(emma.iter_snps(), 3000)])  # 94 strains, 0/1
for kind in ("standardized", "ibs"):
    print(kind, kinship_scale(calculate_kinship(G_hmdp, kind=kind)))
```

pylmm 1.0.0 had two conventions (the library imputed before standardising, the command line did not:
mean diagonal 1.000 vs 0.987 on the bundled BED data) and counted skipped SNPs in `m`, shrinking K by the
share of monomorphic SNPs (7.5% on the HMDP strains); bug 8. `calculate_kinship(W, legacy=True)` and
`pylmm kinship --legacy-kinship` reproduce each 1.0.0 K exactly.

* The `.id` sidecar is what lets `pylmm gwas` (and `PlinkReader(k_file=...)`) align K to the genotype
  file by ID. Without it K's rows are *assumed* to be in genotype order: `pylmm gwas --bfile`/`--tfile`
  warns, the library's `read_kinship` does not (it only checks that K is n × n). For a kinship computed
  elsewhere, write `K.id` yourself: `pylmm.samples.write_id_file("K.id", ids)` or
  `plink_tools.write_kinship("K", K, ids)`.
* EMMA genotypes have no IDs, so with `--emma-geno` (or `type="emma"`) K's rows are always the genotype
  columns in order: no `.id` is needed and none is warned about; a `K.id` that exists is ignored, with a
  warning. Check the order yourself (keep the strain list you wrote the columns from).
* `calculate_kinship(W, center=True)` returns a centred K scaled so that `trace(P K P) = n - 1`.
* With IDs, `PlinkReader(prefix, k_file="K")` reads K, reorders it to the genotype file, and drops
  genotype individuals that are not in it from `indivs`, `phenos` and the genotype vectors.

### Kinship files of GCTA and plink2

`pylmm.kinship_io` reads and writes GCTA's binary GRM (`gcta --make-grm`, `plink2 --make-grm-bin`:
`P.grm.bin`, the lower triangle in float32; `P.grm.N.bin`, SNP counts; `P.grm.id`, `FID IID`) and reads
plink2's `--make-rel` files (`P.rel` text, `P.rel.bin` float64 or float32, square or triangle, with
`P.rel.id`). Both carry their sample IDs, so they align by ID like a kinship with a `.id` file:
`PlinkReader(k_file="P.grm.bin")` or `k_file="P.rel"`, and on the command line `pylmm gwas --grm P` or
`--kinship P.rel`. `pylmm kinship --out-format grm -o P` writes a GRM.

A GRM holds K in **float32**, about 7 significant digits: what you read back is K rounded
(`kinship_io.round_like_grm(K)`), and a scan on it matches the scan on the float64 text kinship to
rounding (|Δt| about 1e-7 on synthetic data), not bit for bit. pylmm's own text kinship
(`pylmm kinship -o K`) keeps every digit; use it when you need the exact K. A plink2 *text* `.rel` is
the least precise of all: plink2 prints it with 6 significant digits (`%g`), so scans on it move by
|Δt| up to about 1e-5. From plink2, prefer `--make-rel bin` (float64, exact) or `--make-grm-bin`.

```python
from pylmm import kinship_io

kinship_io.write_grm("mini", K_2000, ids, n_snps=2000)   # mini.grm.bin, .grm.N.bin, .grm.id
K_grm, grm_ids = kinship_io.read_grm("mini")
assert grm_ids == ids and np.array_equal(K_grm, kinship_io.round_like_grm(K_2000))
print("max relative rounding %.1e" % np.max(np.abs(K_grm - K_2000) / np.abs(K_2000)))
```

## 5. The null fit and heritability

A simulated phenotype with a known heritability makes the numbers checkable: `u ~ N(0, 0.6 K)` plus one
causal SNP. (The `simulate-lmm-data` skill does this with the `sims/` framework.)

```python
from scipy import linalg
from pylmm.lmm import LMM

kva, kve = linalg.eigh(K)                  # reuse it: every LMM(Y, K) would recompute it
rng = np.random.default_rng(1)
u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(n)) * np.sqrt(0.6)
x_causal = W[:, 100]                       # a standardised SNP
Y = 10 + 0.8 * x_causal + u + rng.normal(scale=np.sqrt(0.4), size=n)

model = LMM(Y, K, kva, kve)                # X0 defaults to an intercept
h, beta, sigma, ll = model.fit()           # REML
print("h=%.3f  s2=%.3f  intercept=%.2f  REML LL=%.1f" % (h, sigma, beta[0, 0], ll))
assert 0.4 < h < 0.8
```

* `h` is a numpy float; `sigma` (the total variance `s2`) and `ll` are floats; `beta` is a (q, 1) array.
  The genetic and residual variances are `h * sigma` and `(1 - h) * sigma`.
* After `fit`, the model keeps `opt_h`, `opt_sigma`, `opt_beta`, `opt_ll`, and the log-likelihood on the
  search grid, `ll_grid` over `h_grid` (0, 0.01, ..., 1).
* `fit(reml=False)` is ML, with `sigma = Q / n`. Prefer REML: with a centred kinship built from more SNPs
  than individuals the ML likelihood has no maximum (it grows towards h = 1; bug 17 in the simulation
  studies), and `fit` warns.
* `h` is searched on the closed interval [0, 1]: every local maximum of the grid, the ends included, is
  refined and the best is kept. 0 and 1 are boundary estimates ("no K variance detected", "no residual
  variance detected"), not precise values. At h ≈ 1 (residual variance negligible next to K's) with a rank-deficient K (replicate animals of
  inbred strains) `fit` warns: the Wald test then overstates the information. ĥ does not depend on how
  the covariates are scaled or centred (the fit works on an orthonormal basis of them).
* `LMM(..., legacy_fit=True)` (and `gwas(..., legacy_fit=True)`, `pylmm gwas --legacy-fit`) fits as
  pylmm 1.0.0 did: `h` on 0, ..., 0.99 only (a true h² ≥ 0.98 reads 0.99), the first local maximum, not
  the best, and `sigma = Q / (n - p)` under ML. Use it to reproduce 1.0.0 results.

**Is `h` comparable with a published h²?** `h` is the share of `s2` on the scale of K. It equals the
proportion of phenotypic variance explained only if K is scaled so that the average variance it implies is
1 (Gower's centring: `trace(P K P) / (n - 1) = 1`, `P = I - 11'/n`); a mean diagonal of 1 is the usual
approximation. A kinship from another tool, or another kind (`ibs`), may not be (`kinship_scale`, section
4). Dividing K by
`c` changes `h` exactly as `h' = h c / (h c + 1 - h)`, and changes no test statistic:

```python
P = np.eye(n) - 1.0 / n
c = np.trace(P @ K @ P) / (n - 1)          # 1 for a Gower-scaled K
h_gower = LMM(Y, K / c).fit()[0]
print("c=%.4f  h=%.4f  h(K/c)=%.4f  predicted=%.4f" % (c, h, h_gower, h * c / (h * c + 1 - h)))
assert abs(h_gower - h * c / (h * c + 1 - h)) < 1e-4
```

The `estimate-heritability` skill covers putting `h` on another tool's kinship scale (e.g. a GCTA GRM with
diagonal ≈ 2 in inbred lines), PVE, standard errors and profile-likelihood intervals for `h`.

## 6. Testing one SNP

```python
t, p, b, b_var = model.association(x_causal, return_beta=True)
print("t=%.2f  p=%.2e  beta=%.3f  se=%.3f" % (t, p, b, np.sqrt(b_var)))
assert p < 1e-6
t_null, p_null = model.association(W[:, 1500])   # t and p only
```

* `association` uses the `h` of the last `fit` (or `h=`). It returns numpy floats; `b_var` is a float.
* The SNP may be `(n,)` or `(n, 1)`, over the same individuals as `Y`, and must be complete: a `nan` makes
  scipy raise `ValueError: array must not contain infs or NaNs` (use `gwas`, section 8).
* `beta` is per unit of the column you pass: here per SD of the standardised genotype.
* `association` does not check that the SNP is estimable (not constant, not collinear with `X0`):
  `pylmm.lmm.snp_estimable(x, X0)` does, as `gwas` does for you.
* `fit(X=x)` followed by `association(x)` is the per-SNP refit.

`association_batch` runs the same test on many complete SNPs at once: it rotates the whole block with
one matrix product and fits each SNP in closed form against the null model, so it is many times faster
than a loop over `association`. It returns arrays `t, p, beta, beta_var`. For well-conditioned covariates
the numbers are those of `association` up to rounding (within about 1e-9 relative in the tests). With a
badly scaled covariate next to the intercept (a raw date such as 20240131, or a year column) `association`,
which inverts `X'SX`, loses precision, and the two can differ by up to a few 1e-6 relative; the batched
values are then the more accurate ones (against a 40-digit reference: 3e-11 vs 3e-7 off). Centring or
scaling such covariates (`(c - c.mean()) / c.std()`) changes neither the model nor the test statistics, and
makes both paths accurate. Like `association` it does not check estimability or
handle `nan`.

```python
res = model.association_batch(W[:, 1490:1510])
t_loop = [model.association(W[:, j])[0] for j in range(1490, 1510)]
assert np.allclose(res.t, t_loop, rtol=1e-9)
print("se of the first:", np.sqrt(res.beta_var[0]))
```

## 7. Scans with `gwas`

`gwas(Y, X, K)` tests every column of `X` and returns two lists, t statistics and p-values (`nan` for a SNP
that cannot be tested: constant, collinear with the covariates, all missing).

```python
from pylmm.lmm import gwas

ts, ps = gwas(Y, W, K, kva, kve)           # null fit once, REML
ps = np.asarray(ps)
top = np.argsort(ps)[:3]
print([(snp_names[j], "%.1e" % ps[j]) for j in top])
assert ps[100] < 1e-10                   # the causal SNP; in this inbred panel SNPs in LD with it
                                         # (and with the polygenic effect) can rank even higher
```

**Covariates and the intercept.** `X0` is used exactly as given. `read_covariates` returns the covariate
columns only, so add the intercept yourself, or the model has none and the test statistics change, often
a lot. (`pylmm gwas --covar` adds it for you; `--no-intercept` turns that off.)

```python
X0 = np.column_stack([np.ones(n), cov])    # intercept + sex + age
keep = ~np.isnan(cov).any(axis=1)          # gwas drops missing Y, not missing covariates
ts_cov, ps_cov = gwas(Y[keep], W[keep][:, :200], K[np.ix_(keep, keep)], X0=X0[keep])
ts_noint, _ = gwas(Y[keep], W[keep][:, :200], K[np.ix_(keep, keep)], X0=cov[keep])  # WRONG: no intercept
print("max |t| change without the intercept: %.2f" % np.nanmax(np.abs(np.subtract(ts_cov, ts_noint))))
```

`gwas` and `LMM` drop individuals with a missing `Y` themselves (and recompute the eigendecomposition),
but `X0` must be complete on the individuals with a phenotype, so drop or impute missing covariates first,
as above. Other options: `reml=False` (ML everywhere), `refit=True`, and `ids=` / `kinship_ids=` to have
`gwas` check that Y and K list the same individuals in the same order.

**Speed.** `gwas` tests the complete SNPs (no `nan`) in blocks of `block_size=1024` with
`LMM.association_batch` (section 6); SNPs with missing calls, and every SNP with `refit=True`, are tested
one at a time. `batch=False` tests every SNP one at a time, the code path of pylmm 1.0.0: slower, and the
same numbers up to rounding for well-conditioned covariates (see section 6 for badly scaled ones). A block
takes about 6-8 arrays of `n * block_size` floats. `pylmm gwas --block-size N` and `--no-batch` do the same
on the command line, where the genotypes are also read a block at a time (`iter_blocks`, below), as in
`pylmm kinship`.

```python
ts_one, _ = gwas(Y, W[:, :200], K, kva, kve, batch=False)
assert np.allclose(ts[:200], ts_one, rtol=1e-9, equal_nan=True)
```

**Leave one chromosome out (LOCO).** A K built from the tested SNPs partly explains them (proximal
contamination): the polygenic term absorbs some of a causal SNP's effect, and with a polygenic trait the
whole scan is deflated (λ_GC well below 1 in inbred panels). `pylmm.loco` tests the SNPs of chromosome
`c` with `K_{-c}`, the kinship of the other chromosomes (GCTA `--mlma-loco`; `pylmm gwas --loco` on the
command line). `LocoKinship` builds every `K_{-c}` in one pass over the **raw** genotypes, for any
`kind`, keeping one n × n sum per chromosome (memory: about (chromosomes + 6) × n² × 8 bytes,
`pylmm.loco.loco_memory_bytes`); each `K_{-c}` is
`calculate_kinship` of the other chromosomes' SNPs, up to rounding. `gwas_loco(Y, X, chroms)` fits the
null model on each `K_{-c}` (its own `h`) and tests that chromosome's columns with `gwas`:

```python
from pylmm.loco import LocoKinship, gwas_loco

loco = LocoKinship(n)                                    # standardized, as calculate_kinship
with PlinkReader(BED, norm_genotype=False) as reader:    # raw genotypes, as for calculate_kinship
    chrom = reader.variants.chrom
    for G_block, v in reader.iter_blocks(5000):
        loco.add_block(G_block[:, ::20], v.chrom[::20])  # every 20th SNP, to keep the example fast
print(len(loco.chromosomes), "chromosomes;", loco.snps_used(), "SNPs;",
      loco.snps_used(leave_out="1"), "without chromosome 1")
res = gwas_loco(Y, W, chrom[: W.shape[1]], kinships=loco)   # W: the first 2000 SNPs, chromosome 1
print("null h: %.3f with K of all SNPs, %.3f without chromosome 1" % (h, res.h["1"]))
print("causal SNP: p = %.1e (all-SNP K), %.1e (LOCO)" % (ps[100], res.p_values[100]))
assert res.p_values[100] < ps[100]
```

Each `K_{-c}` has its own eigendecomposition and null fit, so a scan over c chromosomes costs c of them,
and that dominates as n grows (an O(n³) eigendecomposition each): about the cost of the all-SNP scan at
n ≈ 1200 (19 chromosomes), 7 times it at n = 3000 (10 chromosomes). Chromosome labels are normalised
(`normalize_chrom`: `chr1`, `01` and `1` are one chromosome), and with `kinships=` every chromosome of
`chroms` must be in them (an error otherwise). `loco.kinship(c)` returns `K_{-c}` for your own loops
(`loco.kinship()`: all SNPs). The `custom-association-analysis` skill shows the by-hand equivalent.

Under LOCO, λ_GC of a polygenic trait is above 1 without any confounding: the SNPs linked to the tested chromosome's own causal variants are no longer explained by K (S8: 1.1-1.5 on the HMDP strains, up to 2.8 on a synthetic inbred panel, 1.5 on the README mice at h² = 0.5). It is not a calibration check there, and **genomic control must never be applied to a LOCO scan**; check calibration on a null (permuted or simulated) phenotype instead. `sims/studies/s8_loco.md` has the calibration (on chromosomes without causal
variants and on null phenotypes) and the power LOCO buys: large in inbred panels, small for unlinked
outbred SNPs.

## 8. Missing genotypes

* `gwas` handles a SNP with missing calls (`nan`) by testing it on the called individuals only (a new LMM
  on that subset, with its own eigendecomposition: slower).
* `LMM.association` does not; impute first, or use `gwas`.
* Genotypes read with `norm_genotype=True` (the default) are already mean-imputed, so they have no `nan`
  (except an all-missing SNP, returned as all `nan`, and a constant SNP, returned as zeros: both untestable).
* `pylmm gwas` mean-imputes by default; `--remove-missing-genotypes` tests each SNP on its called
  individuals instead.

```python
X = W[:, :5].copy()
X[:50, 2] = np.nan                         # 50 missing calls in SNP 2
ts_m, ps_m = gwas(Y, X, K, kva, kve)
print(["%.3f" % t for t in ts_m])
assert np.all(np.isfinite(ts_m))
```

## 9. The command line

`pylmm gwas` is the same model as `gwas`, reading files. Write the simulated phenotype (and the fake one)
to a phenotype file with a header, and the covariates, then scan the `mini` file set of section 4:

```python
plink_tools.write_pheno("sim.pheno", ids, np.column_stack([Y, Y_fake]), ["trait", "noise"])
plink_tools.write_pheno("sim.covar", ids, cov, ["sex", "age"])
```

```bash
pylmm gwas --bfile mini --kinship data/snps.132k.clean.noX.pylmm.kin \
    --pheno sim.pheno --pheno-name trait --covar sim.covar --covar-name sex age \
    -o trait.tsv -v 2> trait.log
grep "null model" trait.log
head -3 trait.tsv
```

* `--kinship K` uses `K.id`, if it exists, to align K by ID; without it, it warns and assumes genotype
  order. With `--emma-geno` there are no IDs: K is in genotype order, with no warning (and `K.id`, if
  present, is ignored with a warning). `--grm P` reads a GCTA GRM instead, and `--kinship P.rel` a plink2
  one, both with their IDs (section 4).
* `--loco` (instead of `--kinship`/`--grm`) builds `K_{-c}` for every chromosome from the genotypes and
  tests each chromosome's SNPs with its own (section 7): one extra pass over the genotypes, one
  eigendecomposition and null fit per chromosome (logged with `-v`). It needs the chromosomes (the `.bim`
  or `.map`; for `--emma-geno`, a marker map with `--emma-map`, e.g. `data/hmdp.liver.snps.map.tsv`), each
  chromosome's SNPs together in the file, and at least two and at most `--loco-max-chroms` (default 100)
  chromosomes; `--kinship-type` picks the kind. The output columns and order are the usual ones. It logs
  its memory estimate first, about (chromosomes + 6) × n² × 8 bytes. The genotypes are read in full for
  the kinships before the scan, so a genotype read error (a truncated BED, a malformed `.bim` line) stops
  the run before any output, where the standard scan writes the SNPs before the error. Do not apply
  genomic control to its output (λ_GC of a polygenic trait is above 1 by design; section 7). LOCO is
  opt-in: without `--kinship`, `--grm` or `--loco` the command is a usage error that names all three.
* The intercept is added unless `--no-intercept`; individuals with a missing phenotype or covariate are
  dropped (with a warning).
* `--pheno-col N` is 1-based (like plink2 `--pheno-col-nums`); `--pheno-name` picks by header name.
* `--pfile P` reads a PLINK 2 PGEN file set instead of `--bfile` (the `pgen` extra; dosages included,
  `--hardcalls` for the hard calls only); so does `pylmm kinship --pfile P`. The effect allele is REF.
* `--id-match iid` matches on IID alone; `--ml`, `--refit`, `--remove-missing-genotypes` as in the
  library; `--eigen` reuses a decomposition (section 11).
* Exit code 0 on success, 2 for a usage or input error (one `pylmm: error: ...` line), 1 for an unexpected
  error (rerun with `--debug` for the traceback).
* `pylmm gwas --help` lists every option, with examples.

## 10. Reading the results

The output is tab-separated, one line per SNP, in `.bim` order:

| column | meaning |
|---|---|
| `SNP_ID` | `.bim` column 2 |
| `BETA` | the SNP effect, **per SD of the standardised genotype** of the A2 allele (`.bim` column 6) |
| `BETA_SD` | the **standard error** of `BETA` (despite the name) |
| `F_STAT` | the signed **Wald t statistic**, `BETA / BETA_SD` (despite the name; t² is the F) |
| `P_VALUE` | two-sided, from t(n - q - 1) |

`nan` in all four value columns marks a SNP that cannot be tested.

```python
res = np.genfromtxt("trait.tsv", names=True, dtype=None, encoding="utf-8")
assert np.allclose(res["F_STAT"], res["BETA"] / res["BETA_SD"], equal_nan=True)
best = res[np.nanargmin(res["P_VALUE"])]
print(best["SNP_ID"], "%.2e" % best["P_VALUE"])
assert res["P_VALUE"][100] < 1e-10
```

**Per-allele effects.** To compare `BETA` with tools that report the effect per copy of A1 (plink2 `--glm`,
GEMMA, GCTA and REGENIE count A1 = ALT), divide by the SD of the 0/1/2 dosage *over the called
individuals* (the SD the reader standardised by) and flip the sign: `beta_A1 = -BETA / np.nanstd(2 *
g_raw)`, with `g_raw` the `norm_genotype=False` values over the individuals of the genotype file (after
any dropped by the kinship). `BETA_SD` converts the same way; t and p need no conversion (only the sign of
t flips with the allele). Do not compare raw `BETA` values between runs with and without
`--remove-missing-genotypes` (section 12).

The same conversion, checked with the library on the causal SNP (`x_causal` is the standardised SNP 100,
`model` the fit of section 5):

```python
with PlinkReader(BED, norm_genotype=False) as reader:
    g_raw = next(itertools.islice(reader.iter_snps(), 100, None))[0]
sd_called = np.nanstd(2 * g_raw)           # SD of the dosage over the called individuals
dosage_a1 = 2 - 2 * np.where(np.isnan(g_raw), np.nanmean(g_raw), g_raw)   # copies of A1, mean-imputed
_, _, b_std, _ = model.association(x_causal, return_beta=True)
_, _, b_a1, _ = model.association(dosage_a1, return_beta=True)
print("per SD of A2: %.4f   per copy of A1: %.4f   -b_std / sd: %.4f" % (b_std, b_a1, -b_std / sd_called))
assert np.isclose(b_a1, -b_std / sd_called)
```

The `interpret-gwas-results` skill covers λ_GC, QQ plots, thresholds and when to distrust a scan.

## 11. Performance

* The cost is one O(n³) eigendecomposition of K plus O(n²) per SNP (a rotation). n of a few thousand is
  quick; K itself is 8 n² bytes (3.2 GB at n = 20,000), the practical limit.
* Compute the eigendecomposition once and pass it: `LMM(Y, K, kva, kve)`, `gwas(Y, X, K, kva, kve)`. It is
  ignored (recomputed) when `Y` has missing values, because K is then subset.
* `pylmm kinship --eigen-npz eig.npz` writes it for `pylmm gwas --eigen eig.npz`, with the sample IDs of
  its rows. `pylmm gwas` matches them to the tested individuals: a sample it lacks is an error (the file
  belongs to another kinship), a different order is fixed, and if individuals were dropped (a missing
  phenotype, not in the kinship) it is ignored and recomputed, with a warning. A file without IDs
  (pylmm 1.0.0's, or `--eigen-text` without its `.id`) is used only when no individual is dropped or
  reordered, with a warning.
* SNPs with missing calls under `--remove-missing-genotypes` (or library `gwas`) each cost an
  eigendecomposition; `--refit` costs a likelihood search per SNP.
* Process SNPs in blocks with `reader.iter_blocks(block_size)` rather than building a full genotype matrix;
  for a BED file it also decodes 2-6x faster than `iter_snps` (unstandardised; about 2x standardised, on
  1219 mice; `sims/bench/bed_blocks.py` measures it). `pylmm gwas` and
  `pylmm kinship` read their genotypes this way.

```bash
pylmm kinship --bfile mini -o mini.kin --eigen-npz mini.eig.npz -v
pylmm gwas --bfile mini --kinship mini.kin --eigen mini.eig.npz --pheno sim.pheno -o fast.tsv -v
```

## 12. Pitfalls

1. **No intercept with library covariates.** `X0` is used as given; prepend `np.ones(n)`.
2. **Kinship order.** A kinship without a `.id` file is assumed to be in genotype order (the CLI warns
   for PLINK input; EMMA input and the library do not). Write the `.id` file whenever K comes from
   elsewhere, and check `samples.read_id_file(...) == reader.indivs`.
3. **FID mismatches.** Phenotype rows are matched on `(FID, IID)`. If your phenotype file uses other FIDs
   than the `.fam` (e.g. `0`, or plink2's `--double-id`), every individual is "missing"; use
   `--id-match iid` / `id_match="iid"`.
4. **`#FID` headers** (plink2 style) are not recognised as headers; write `FID IID ...`.
5. **`-9` in covariates** is a value, not missing (only `NA` is). In phenotypes `-9` is missing.
6. **Effect allele and scale.** BETA is per SD of the A2 allele (REF for plink2-written files, the
   *major* allele for plink 1.9 without `--keep-allele-order`), opposite in sign to plink2 `--glm`.
   With `--remove-missing-genotypes`, SNPs without missing calls are tested on the raw 0–1 scale and SNPs
   with missing calls on the standardised scale, so BETA is not comparable across SNPs (t and p are fine).
7. **`h` at a boundary.** 1 means no residual variance detected, 0 none due to K; neither is a precise
   estimate (use a profile-likelihood interval, `estimate-heritability` skill). ML can run to 1 for a
   centred K from more SNPs than individuals (`fit` warns); use REML. (pylmm 1.0.0 and `--legacy-fit`
   cap `h` at 0.99.)
8. **K scale.** `h` depends on how K is scaled (section 5); test statistics do not.
9. **Inbred panels.** Replicate animals of one strain share a genotype: K is rank-deficient (the example
   K has rank about 110 for 1219 mice), the effective number of independent tests is small, and λ_GC from
   one scan is noisy. Individual-level vs strain-mean heritability and what to expect: the
   `estimate-heritability` skill, section 7. Data preparation (strain aliases, duplicate strain columns,
   strain → animal expansion, imputing missing calls over strains, not animals): the
   `inbred-strain-panels` skill.
10. **`association` with `nan`** fails; `gwas` handles missing calls.
11. **Imputation set.** When K drops individuals, genotypes are standardised over the kept individuals, but
    individuals dropped for a missing phenotype or covariate still count in the mean and SD (|Δt| ≤ 0.024
    in our check; known, not yet fixed).

## 13. Validated behaviour

pylmm's default path (REML null fit, Wald test) was checked by simulation studies
([docs/phase4-simulations.md](phase4-simulations.md), framework in [sims/README.md](../sims/README.md)):

* **Calibrated null p-values** on unrelated, structured (Balding–Nichols), family and the bundled mouse
  data, for h² > 0: pooled type-I error 0.0495–0.0500 at α = 0.05 and mean χ² 0.99–1.01. At h² = 0 the
  test is mildly conservative (REML puts ĥ at 0).
* **Matches an independent dense REML oracle**: Wald t to 2e-7, ĥ RMS within 0.0033 of the textbook
  optimum for h² ≤ 0.95.
* **Power ≥ OLS**, much higher under population structure (e.g. 0.94 vs 0.15 at h² = 0.5, 5% variance
  explained, Balding–Nichols), and OLS is badly inflated there.
* **The h search** (`sims/studies/s7_numerics.py`): ĥ on [0, 1] reaches the oracle's maximum in every
  simulated fit, including true h² of 0.98 and 1 (1.0.0 capped it at 0.99 and could keep a lower local
  maximum).
* **Known limits** (see *Known caveats* in AGENTS.md): ML unidentified with a centred K from m > n
  SNPs. (The kinship normalisation, bug 8, is fixed: section 4.)
* The golden tests (`tests/golden/`) pin every number of the original 2013 code, run with
  `legacy_fit=True` / `--legacy-fit`; `pylmm gwas --legacy-fit --no-batch` writes pylmm 1.0.0's output byte
  for byte (`tests/test_legacy_fit.py`).

**Against published results** ([replication.md](replication.md)): an agent using only these docs
and the skills reanalysed two published mouse GWAS from their original phenotypes. Farber et al. 2011
(bone mineral density): **partially replicated**. All 5 published loci have a signal in the published ±1.3 Mb interval, 3 at the published position (median
−log10 p 0.47 below published); 1/5 pass Bonferroni and 3/5 the paper's threshold. Bennett et al. 2010
(HDL): **replicated**, same lead SNP, −log10 p 11.5 vs 9.4 published.
Closest method: an IBS-style kinship with a per-SNP REML refit. No numerical bug was found.
