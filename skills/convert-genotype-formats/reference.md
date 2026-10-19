# Conversion reference: recipes, caveats and how they were checked

Companion to [SKILL.md](SKILL.md). Research background: [docs/research/io-formats.md](../../docs/research/io-formats.md).
The commands below are not run in CI (they need external tools). In Phase 5c (October 2026) each recipe
was run on a small test VCF (8 samples; a multiallelic site, a half-call, an `R2` INFO field, a `DS`
FORMAT field) with **plink2 v2.0.0-a.7.13 (15 Oct 2026)**, **bcftools 1.24**, **bio2zarr 0.2.1** and
**vcztools 0.2.0**, and the resulting BED read with pylmm. Results are marked *checked*; anything taken
from documentation only is marked *docs only*.

## PLINK 1 vs PLINK 2

| | PLINK 1 (BED/BIM/FAM) | PLINK 2 (PGEN/PVAR/PSAM) |
|---|---|---|
| genotypes | hard calls, 2 bits | hard calls + dosages (1/16384), phase, multiallelic |
| alleles | `.bim` A1 (col 5), A2 (col 6); no REF/ALT | `.pvar` REF and ALT explicit |
| sample IDs | `.fam` FID IID (both required) | `.psam` `#IID` or `#FID IID` header |
| pylmm | **reads it** (counts A2) | convert to BED (lossy for dosages) |

* **plink2 `--make-bed` writes A1 = ALT, A2 = REF** (*checked*: `.bim` of a VCF with REF `A`, ALT `G`
  reads `G A`). pylmm counts A2, so its effect allele is REF. A VCF `0/0` (REF/REF) reads as 1.0 in
  pylmm (*checked*).
* plink2 reading a BED treats A2 as REF (provisionally) and A1 as ALT (*checked* with `--freq`).
* **plink 1.9** reorders alleles so A1 is the minor allele unless `--keep-allele-order` (*docs only*).
  Effects then refer to the major allele in pylmm and can flip between datasets.
* To fix REF from a reference genome: `plink2 ... --ref-from-fa --fa ref.fa` (*docs only*).

## VCF/BCF (called genotypes) → BED (*checked*)

```bash skip-ci: needs bcftools and plink2
bcftools norm -m -any -Ob -o split.bcf in.vcf.gz && bcftools index split.bcf
plink2 --bcf split.bcf \
  --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 100 missing \
  --vcf-half-call missing --max-alleles 2 \
  --double-id \
  --autosome --snps-only just-acgt --maf 0.01 --geno 0.05 \
  --make-bed --out geno
```

* `bcftools norm -m -any` splits multiallelic records into biallelic ones (*checked*: `A>G,T` became two
  records). Add `-f ref.fa` to also left-align indels (needs the matching FASTA). `bcftools view -m2 -M2
  -v snps` keeps biallelic SNPs only instead (*checked*).
* Without splitting or `--max-alleles 2`, `--make-bed` stops: `Error: geno.bim cannot contain multiallelic
  variants.` (*checked*). `--max-alleles 2` drops them (`1 excluded by --max-alleles`).
* The default `--vcf-half-call` is `error`: `Error: Line 9 of --vcf file has a GT half-call.` (*checked*).
  `missing` sets the call to missing.
* `--set-all-var-ids '@:#:$r:$a'` names SNPs `chr:pos:ref:alt` (single quotes stop bash expanding `$r`).
  Without it a VCF with `ID = .` gives `.` for every SNP.
* `--double-id`: FID = IID. Without it plink2 writes FID `0` (*checked*). Match the phenotype file to
  whichever you choose, or use `pylmm gwas --id-match iid`.
* `--geno 0.05` / `--maf 0.01` are QC filters; adjust for small or inbred samples (one missing call in 8
  samples already exceeds `--geno 0.05`).
* Phase is dropped; X/Y/MT are dropped by `--autosome` (pylmm has no sex-chromosome model).

## Imputed VCF (DS) → PGEN → BED (*checked*)

```bash skip-ci: needs plink2
plink2 --bcf split.bcf dosage=DS --vcf-half-call missing --extract-if-info 'R2 >= 0.3' \
  --set-all-var-ids '@:#:$r:$a' --double-id --make-pgen --out imp
plink2 --pfile imp --hard-call-threshold 0.1 --make-bed --out imp_hc
```

* **`--extract-if-info` is a valid plink2 flag** (alias `--extract-if`; *checked* against the plink2 docs
  and by running it): `'R2 >= 0.3'` as one quoted expression and `R2 '>=' 0.3` as three tokens both work
  (`2 excluded by --extract-if-info`). Expressions can combine predicates with `&&`, `||`, `!` and
  parentheses; quote them for the shell. Use `INFO`, `DR2` or `R2` per your imputation server.
* **`dosage=DS` refuses multiallelic records, even with `--max-alleles 2`**: `Error: --vcf multiallelic
  dosage import is under development.` (*checked*). Split (`bcftools norm -m -any`) or drop
  (`bcftools view -M2`) them first.
* `dosage=HDS` for Minimac4 haplotype dosages; `dosage=GP` reads probabilities (*docs only*).
* BED stores hard calls: with `--hard-call-threshold 0.1`, dosages more than 0.1 from 0/1/2 become
  missing (and 0.5 / 1.5 always do). pylmm cannot read dosages today; a PGEN reader is planned
  (Phase 6.2d). Alternatively, read dosages with any Python library and pass the matrix to
  `pylmm.lmm.gwas` directly (it takes any numeric coding).

## BGEN → PGEN / BED (*checked*)

```bash skip-ci: needs plink2
plink2 --bgen in.bgen ref-first --sample in.sample --set-all-var-ids '@:#:$r:$a' --make-pgen --out g
plink2 --pfile g --max-alleles 2 --make-bed --out g_hc         # hard calls (lossy, as above)
```

* **plink2 now requires a REF/ALT mode**: without one it stops with `--bgen now requires a REF/ALT mode
  ('ref-first', 'ref-last', or 'ref-unknown')` and notes that raw biobank files are typically
  `ref-first`, older and PLINK-exported files `ref-last` (*checked*; the research note in
  `docs/research/io-formats.md` predates this). The wrong mode swaps REF and ALT for every variant
  (*checked*: reading a `ref-last` file as `ref-first` swapped them).
* `--sample` is optional for BGEN v1.2+ files with embedded sample IDs. Add `--keep`, `--extract`,
  `--maf` to subset; convert one chromosome at a time to bound memory.

## PGEN → BED

```bash skip-ci: needs plink2
plink2 --pfile in --max-alleles 2 --make-bed --out out
```

## VCF Zarr → BED (*checked*)

```bash skip-ci: needs bio2zarr and vcztools
uvx --from 'bio2zarr[vcf]' vcf2zarr convert in.vcf.gz in.vcz     # needs the [vcf] extra (cyvcf2)
uvx vcztools view-plink --max-alleles 2 -o geno in.vcz
```

* **`vcztools view-plink -o/--output STEM` and `-M/--max-alleles INT` are the current flags** (*checked*
  with `vcztools view-plink --help`, version 0.2.0). `-o` is required and is taken verbatim (writes
  `STEM.bed/.bim/.fam`); `--no-bim` / `--no-fam` skip the sidecars. Multiallelic variants are rejected
  unless `-M 2`. Sample/region selection mirrors `bcftools view` (`-s`, `-S`, `-r`, `-R`, `-t`, `-T`,
  `-i`, `-e`, `-v`, `-V`, `-m`, `-M`).
* A1 = ALT, A2 = REF; the `.bed` was **byte-identical** to `plink2 --vcf ... --make-bed` on the test
  data (*checked* with `cmp`). FID = IID in the `.fam`.
* vcztools has no ID option: VCF IDs of `.` stay `.`. Rename in the `.bim`, e.g.
  `awk 'BEGIN{OFS="\t"} {$2=$1":"$4":"$6":"$5; print}' geno.bim > tmp && mv tmp geno.bim`.
* vcf2zarr needs a bgzipped (`.vcf.gz`) input; `vcztools` needs Python 3.11–3.13 (`uvx --python 3.12`).
* In Python, `ds.call_genotype.sum(-1)` of a VCZ dataset (sgkit/xarray) gives ALT counts (n_variants ×
  n_samples, negative = missing) to pass to `gwas` directly, transposed and with missing set to `nan`.

## Hail MatrixTable → BED (*docs only*)

```python skip-ci: needs Hail (a JVM)
mt = hl.split_multi_hts(mt)
hl.export_plink(mt, "geno", ind_id=mt.s, fam_id=mt.s,
                varid=hl.variant_str(mt.locus, mt.alleles))
```

* Signature: `hl.export_plink(dataset, output, call=None, fam_id=None, ind_id=None, pat_id=None,
  mat_id=None, is_female=None, pheno=None, varid=None, cm_position=None)`. It requires no multiallelic
  variants (split or filter first), diploid **unphased** calls, rows keyed by `locus, alleles` and a
  string column key (Hail docs).
* Not run here (needs a JVM/Spark). Hail's docs do not state which allele becomes A1: **check the `.bim`**
  against `mt.alleles` for a few variants before interpreting effect signs.

## Kinship files from other tools

| tool | output | read into pylmm |
|---|---|---|
| GCTA `--make-grm`, plink2 `--make-grm-bin` | `.grm.bin` (float32 lower triangle with diagonal, row-major), `.grm.N.bin`, `.grm.id` (FID IID, no header; *checked* for plink2) | `read_gcta_grm` in SKILL.md, then `plink_tools.write_kinship` |
| plink2 `--make-rel square` | `.rel` text n × n, `.rel.id` with a `#FID IID` header (*checked*) | `np.loadtxt` both (it skips `#` lines) |
| GEMMA `-gk 1` / `-gk 2` | `output/*.cXX.txt` / `*.sXX.txt`, n × n text in `.fam` order | `np.loadtxt`; write the `.fam` IDs as `.id` |

On the bundled 1219 mice, a plink2 GRM (5000 SNPs) has mean diagonal 1.99 (inbred: 1 + F ≈ 2), while
pylmm's is ≈ 1. Test statistics are invariant to the scale; h² is not.

## Phenotype files from data frames

```python needs=pandas
import pandas as pd

# from a file: pd.read_csv(path, dtype={"FID": str, "IID": str}) keeps IDs such as 007 intact
df = pd.DataFrame({"FID": ["f1", "f1", "f2"], "IID": ["007", "008", "101"],
                   "weight": [20.5, None, 22.0], "sex": ["F", "M", "F"]})
df[["FID", "IID", "weight"]].to_csv("pheno.txt", sep="\t", index=False, na_rep="NA")
print(open("pheno.txt").read())
assert open("pheno.txt").read().splitlines()[2] == "f1\t008\tNA"
```

The header must start `FID IID` (not `#FID`); `NA` for missing; numeric values only. IDs as strings
exactly as in the `.fam` (watch leading zeros and float-formatted IDs like `101.0`).

## Downloading public data

* Prefer the provider's PLINK or VCF release; record the URL, the release date and the genome build in a
  README next to the files.
* `curl -fLO URL` (`-f` fails on HTTP errors instead of saving an error page); check checksums.
* Look at the files before converting: `zcat in.vcf.gz | head -n 50`, `head geno.bim geno.fam`, the number
  of samples and variants, the chromosome names (`1` vs `chr1`), and whether sample IDs match the
  phenotype source.
* Archive formats (`.tar.gz`, `.zip`): extract into an empty directory and list it before use.
