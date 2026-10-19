---
name: convert-genotype-formats
description: Convert genotype data to the PLINK 1 BED/BIM/FAM files pylmm reads - from VCF/BCF (bcftools + plink2), imputed VCF and BGEN (via PGEN, with the dosage and REF/ALT caveats), PGEN, VCF Zarr (vcztools view-plink), Hail MatrixTables and plain genotype tables (numpy, with the bundled plink_tools.py) - and prepare the other inputs, phenotype and covariate files (FID IID header, NA missing codes, matching IDs) and a kinship from another tool with its .id file (GCTA .grm.bin, plink2 .rel). Explains PLINK 1 vs PLINK 2 allele conventions and which allele pylmm counts. Use when the data is not already a PLINK BED file set, when downloading public genotype data for pylmm, or when preparing phenotype, covariate or kinship files.
---

# Convert genotype data for pylmm

pylmm reads **PLINK 1 BED/BIM/FAM** (SNP-major, biallelic, hard calls), PLINK TPED and EMMA text. Convert
everything else to BED with standard tools. Full recipes, every caveat and how each was verified:
[reference.md](reference.md); background: [docs/research/io-formats.md](../../docs/research/io-formats.md).

Blocks marked `skip-ci` need plink2 (≥ 2.0 alpha 6; checked with v2.0.0-a.7.13), bcftools (checked
1.24) or vcztools (checked 0.2.0); the other blocks run in CI, in order, from the repository root.

## Which allele pylmm counts

pylmm reads a BED genotype as the **fraction of copies of A2**, the 6th `.bim` column (0, 0.5, 1; nan
missing), and reports `BETA` for that allele (per SD). plink2 writes **A1 = ALT, A2 = REF**, so after a
plink2 conversion pylmm's effect allele is **REF**, the opposite of plink2 `--glm` (which counts A1 =
ALT). plink 1.9 sets A1 to the *minor* allele unless you pass `--keep-allele-order`. After converting,
always look at the `.bim` and record which allele is A2.

## Recipes (summary)

**VCF/BCF, called genotypes → BED**

```bash skip-ci: needs bcftools and plink2
bcftools norm -m -any -Ob -o split.bcf in.vcf.gz && bcftools index split.bcf   # add -f ref.fa to left-align
plink2 --bcf split.bcf --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 100 missing \
  --vcf-half-call missing --max-alleles 2 --double-id \
  --autosome --snps-only just-acgt --maf 0.01 --geno 0.05 --make-bed --out geno
```

**Imputed VCF (DS) → PGEN → BED (hard calls; lossy)**

```bash skip-ci: needs plink2
plink2 --bcf split.bcf dosage=DS --vcf-half-call missing --extract-if-info 'R2 >= 0.3' \
  --set-all-var-ids '@:#:$r:$a' --double-id --make-pgen --out imp
plink2 --pfile imp --hard-call-threshold 0.1 --make-bed --out imp_hc
```

**BGEN → PGEN → BED**

```bash skip-ci: needs plink2
plink2 --bgen in.bgen ref-first --sample in.sample --set-all-var-ids '@:#:$r:$a' --make-pgen --out g
plink2 --pfile g --max-alleles 2 --make-bed --out g_hc
```

**VCF Zarr → BED**

```bash skip-ci: needs vcztools
uvx --from 'bio2zarr[vcf]' vcf2zarr convert in.vcf.gz in.vcz     # only if you start from a VCF
uvx vcztools view-plink --max-alleles 2 -o geno in.vcz            # same .bed as plink2 --make-bed
```

**Hail → BED**: `hl.export_plink(hl.split_multi_hts(mt), "geno", ind_id=mt.s)` (see reference.md).

Key caveats (details in reference.md):

* `--make-bed` errors on multiallelic variants (`.bim cannot contain multiallelic variants`): split them
  (`bcftools norm -m -any`) or drop them (`--max-alleles 2`).
* plink2 stops on a half-call (`./1`) unless `--vcf-half-call missing` (or `haploid`/`reference`).
* `dosage=DS` import rejects multiallelic records even with `--max-alleles 2` ("multiallelic dosage
  import is under development"): split or filter them with bcftools first.
* BED holds hard calls only: with `--hard-call-threshold 0.1`, dosages farther than 0.1 from an integer
  become **missing**. pylmm then mean-imputes them (or drops them with `--remove-missing-genotypes`).
* plink2 now requires a REF/ALT mode for BGEN: `ref-first`, `ref-last` or `ref-unknown` (its message:
  raw UK Biobank files are ref-first, older and PLINK-exported ones more likely ref-last). Check the
  provider's documentation; a wrong choice swaps every allele.
* VCFs without IDs give `.` for every SNP; pylmm would name all SNPs `.`. Use `--set-all-var-ids` (or
  rewrite `.bim` column 2; vcztools has no ID option).
* `--double-id` makes FID = IID; without it plink2 writes FID `0` for VCF samples. Whatever you choose,
  the phenotype file must use the **same FID**, or run `pylmm gwas --id-match iid`.
* Download public data with `curl -LO` (or `wget`), check the published checksum (`shasum -a 256`), and
  note the genome build and the source URL next to the files.

## Plain genotype tables (numpy → BED)

Many public datasets ship genotypes as text tables (one row per SNP or per sample, allele letters or
0/1/2). Convert in numpy and write the BED with `plink_tools.write_bed`, which takes **copies of A2**
(0/1/2, nan = missing). Genotypes read back with `PlinkReader(..., norm_genotype=False)` are fractions
(0/0.5/1): pass `2 * G`, or `coding="fraction"`. Any other value raises `ValueError` (nothing is rounded).

```python
import sys
import numpy as np
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools
from pylmm.plink_io import PlinkReader

# a table of 4 samples x 3 SNPs with allele letters, "NN" = no call
table = {"rs1": ["AA", "AG", "GG", "NN"], "rs2": ["CC", "CC", "CT", "TT"], "rs3": ["GG", "AG", "AA", "AG"]}
a1 = {"rs1": "A", "rs2": "C", "rs3": "A"}            # choose A1/A2 per SNP; pylmm counts A2
a2 = {"rs1": "G", "rs2": "T", "rs3": "G"}
snps = list(table)
G = np.array([[np.nan if gt == "NN" else gt.count(a2[s]) for gt in table[s]] for s in snps]).T
plink_tools.write_bed("table", G, iids=["s1", "s2", "s3", "s4"], snp_ids=snps,
                      chrom=["1", "1", "2"], pos=[100, 200, 50],
                      a1=[a1[s] for s in snps], a2=[a2[s] for s in snps])
with PlinkReader("table", norm_genotype=False) as r:
    back = np.column_stack([g for g, _ in r])
assert np.allclose(back, G / 2, equal_nan=True)       # pylmm reads the A2 fraction
print(open("table.bim").read())
```

For inbred lines coded as one allele per strain (`A`/`G`), use `2 * (allele == a2)`. `write_bed` was
checked against plink2 (`--export A` of its output gives the same counts). `plink_tools.subset_bed` copies
a subset of SNPs (e.g. one chromosome) byte for byte; `read_bim` returns the `.bim` columns.

## Phenotype and covariate files

`FID IID name1 name2 ...`, whitespace-separated. The header's first two fields must be `FID IID` (any
case): a plink2-style `#FID` or `#IID` header is read as data and fails. Missing: `NA` (and `-9` in
phenotypes only). Every value numeric (encode sex as 0/1, one-hot categorical covariates yourself).

```python
from pylmm import samples

# fix a plink2-style header and a text missing code, then check the IDs against the .fam
raw = "#FID\tIID\tweight\tsex\n0\ts1\t20.5\tF\n0\ts2\tNaN\tM\n0\ts3\t22.0\tF\n0\ts4\t19.1\tM\n"
rows = [line.split("\t") for line in raw.strip().split("\n")]
fam = samples.read_id_file("table.fam")                     # [(FID, IID), ...]
fid_of = {iid: fid for fid, iid in fam}
with open("table.pheno", "w") as f:
    f.write("FID IID weight\n")
    for _fid, iid, w, _sex in rows[1:]:
        f.write("%s %s %s\n" % (fid_of[iid], iid, "NA" if w.lower() == "nan" else w))
with open("table.covar", "w") as f:
    f.write("FID IID female\n")
    for _fid, iid, _w, sex in rows[1:]:
        f.write("%s %s %d\n" % (fid_of[iid], iid, sex == "F"))
al = samples.align_samples({"genotypes": fam, "phenotypes": samples.read_sample_table("table.pheno").ids})
assert al.n == 4, al.report()
print(open("table.pheno").read())
```

`plink_tools.write_pheno(path, ids, values, names)` writes such a file from arrays. Check the match before
a long run: `align_samples(...).report()` lists the samples each file lacks.

## A kinship from another tool, and its `.id` file

pylmm's `--kinship` is a plain n × n text matrix; its rows are named by `K.id` (`FID IID` per line).
Without `K.id`, the rows are *assumed* to be in `.fam` order. Convert and write both:

```python
def read_gcta_grm(prefix):
    """GCTA / plink2 --make-grm-bin: float32 lower triangle (row-major, with the diagonal) + .grm.id."""
    ids = [tuple(row[:2]) for row in np.loadtxt(prefix + ".grm.id", dtype=str, ndmin=2)]
    n = len(ids)
    tri = np.fromfile(prefix + ".grm.bin", dtype=np.float32).astype(float)
    assert tri.size == n * (n + 1) // 2, "not a GRM of %d samples" % n
    K = np.zeros((n, n))
    K[np.tril_indices(n)] = tri
    return K + K.T - np.diag(K.diagonal()), ids

# a small GRM in GCTA's layout, to show the round trip
K_true = np.array([[1.0, 0.5, 0.1, 0.0], [0.5, 1.0, 0.0, 0.2], [0.1, 0.0, 1.0, 0.3], [0.0, 0.2, 0.3, 1.0]])
K_true[np.tril_indices(4)].astype(np.float32).tofile("other.grm.bin")
samples.write_id_file("other.grm.id", fam)
K_other, k_ids = read_gcta_grm("other")
assert np.allclose(K_other, K_true)
plink_tools.write_kinship("other.kin", K_other, k_ids)       # other.kin + other.kin.id
```

* plink2 `--make-rel square` writes a text `.rel` (plain n × n) and `.rel.id` with a `#FID IID` header;
  `np.loadtxt` skips the `#` line, so `samples.write_id_file("K.id", ...)` from it works the same way.
* GEMMA `-gk` output is plain n × n text in `.fam` order: write the `.fam` IDs as its `.id`.
* Scales differ: a GCTA-style GRM of inbred lines has diagonal ≈ 1 + F ≈ 2 (≈ 1.99 on the bundled mice),
  pylmm's ≈ 1. Test statistics do not care; h² does (see `estimate-heritability`).

## After converting: checks

```bash
pylmm kinship --bfile table -o table.kin -q
pylmm gwas --bfile table --kinship table.kin --pheno table.pheno -o table.tsv -v 2>&1 | grep -i "individuals\|null model"
```

1. `.bim`: SNP IDs not `.`, A1/A2 as expected, no duplicate IDs; `.fam`: the IDs your phenotype uses.
2. `pylmm gwas -v` reports how many individuals were matched and dropped.
3. Allele frequencies: `np.nanmean(g_raw)` is the A2 frequency; compare a few SNPs with the source.
4. Run a quick simulation through the same pipeline (`simulate-lmm-data`, section 5).
