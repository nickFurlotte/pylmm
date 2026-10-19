# Example data: what it is, where it comes from, checksums

Two example datasets ship with pylmm (~215 MB): a 94-strain mouse panel in EMMA text format (`hmdp.*`)
and a 1219-animal PLINK BED file set with **fake phenotypes** (`snps.132k.clean.noX.*`). The tests, the
README example, `scripts/example.py` and the docs use them. They were added to the repository in
2012–2013 (the `.kin.id` sidecar and the marker map later); this file records what is known about them,
and what was verified, so that analyses on them can say what the rows and columns are.

## The HMDP strain panel (EMMA format)

The Hybrid Mouse Diversity Panel (HMDP): classic inbred strains plus AXB/BXA, BXD and BXH recombinant
inbred strains. **One column per strain, no animals**: these are strain-level genotypes and strain-level
expression values.

| file | what |
|---|---|
| `hmdp.liver.snps` | EMMA genotype matrix: 132,285 rows (SNPs) × 94 columns (strains), whitespace-separated, coded `0`/`1` (inbred: homozygous for one allele or the other), `NA` = missing (2.1% of calls). No SNP names, positions or alleles in the file: see the marker map. |
| `hmdp.liver.snps.map.tsv` | **validated marker map** for `hmdp.liver.snps`, one line per row (below). |
| `hmdp.liver.strains` | the 94 strain names, one per line in double quotes, in column order. |
| `hmdp.liver.exprs` | liver gene-expression values of the same 94 strains: 22,716 rows (transcripts) × 94 columns in the same strain order. No transcript IDs or processing notes are bundled; treat it as example phenotypes, not as an annotated expression dataset. |
| `hmdp.liver.exprs.1` | the first row of `hmdp.liver.exprs` (one phenotype, the `--emma-pheno` example of `scripts/example.py` and the tests). |
| `hmdp.liver.K.npdump` | 94 × 94 kinship, `calculate_kinship` of all 132,285 SNPs (agrees to 2e-13). A Python 2 `ndarray.dump` pickle: `np.load(path, allow_pickle=True, encoding="latin1")`; only load pickles you trust. |
| `hmdp.liver.snps.npdump` | the full genotype matrix as a pickle, **missing calls already mean-imputed** (unlike the text file). |
| `hmdp.liver.snps.1000.npdump` | the first 1000 rows as a pickle, `nan` for missing (equal to the text file). |

**Origin and build.** The rows of `hmdp.liver.snps` are, in order, the rows of the Broad Institute's
merged mouse HapMap / wild-type genotype file, NCBI build 37 (mm9):
<https://data.broadinstitute.org/mouse/hapmap/> → `Merged_Hapmap_WT_Genotypes.txt.gz`
(20,446,478 bytes, last modified 2009-02-20, SHA-256 `dd4f80e9e737635c02a9f2f2e843ee3666a91d9b4ed4903ce453128bb126796f`;
the paper-era URL `http://www.mousehapmap.org/` redirects there over plain HTTP; HTTPS on that hostname
fails). This was not documented when the data was added; the Phase R replication established it
empirically (2026-10-19): **the same number of rows (132,285)**, and across the 27 strains whose names
appear in both files, **122,242 informative rows with 0 discordances** between the 0/1 codes and the
Broad base calls. The other 10,043 rows show one base only in those 27 strains, so they are consistent
but not independently checked. The matrix includes 1,377 X-chromosome rows; autosomes 1–19 otherwise.

**The marker map** `hmdp.liver.snps.map.tsv` (tab-separated, header line) is that join, written by
[`scripts/build_hmdp_marker_map.py`](../scripts/build_hmdp_marker_map.py), which downloads the Broad
file, checks its SHA-256 and the row count, classifies every row and refuses to write a map with a
discordant row:

| column | meaning |
|---|---|
| `row` | 1-based row of `hmdp.liver.snps` (pylmm's EMMA output names it `SNP_<row>`) |
| `chr` | chromosome, `1`–`19` or `X` |
| `bp_b37` | position, NCBI build 37 (mm9); lift over before comparing with mm10/GRCm38 or later results |
| `allele0` | the base coded `0`, or `.` if none of the 27 shared strains carries it |
| `allele1` | the base coded `1`, or `.`; for EMMA input pylmm's BETA is the effect of this allele (per SD) |
| `status` | `concordant` (both bases seen, codes consistent: row identity checked), `uninformative` (one base seen in the shared strains), never `discordant` |

Rows with an unknown allele (`.`, 10,043 rows) are fine for scans; give them placeholder letters when you
write a PLINK `.bim` (`inbred-strain-panels` skill). The Broad file has no rsIDs for most rows, so compare
with published loci by position on build 37. Rebuild the map (needs the network, ~20 MB download, ~5 s):

```bash skip-ci: downloads the Broad mouse HapMap file
uv run python scripts/build_hmdp_marker_map.py
```

**The strain list.** 94 columns, 92 distinct names. `AXB-19/PgnJ` appears **three times** (columns 9,
10 and 12, 1-based), and the three columns are identical, including missing calls. In the Mouse Phenome
Database the corresponding phenotype groups are the coisogenic sublines **AXB19a/PgnJ, AXB19/PgnJ and
AXB19b/PgnJ** (historical labels AXB-18, AXB-19 and AXB-20 in Farber et al. 2011), which share one
genotype vector. Do not drop the duplicates blindly, and do not pick one of three identical columns by
its association: keep the phenotype groups and give each the shared vector (see the
`inbred-strain-panels` skill for the identity audit). Strain names contain `/`, `<`, `>`, `+` and, in
other sources, spaces (`BTBR T+ tf/J`, here `BTBRT<+>tf/J`); MPD has renamed some strains between exports.

```text
129X1/SvJ A/J AKR/J AXB-1/PgnJ AXB-10/PgnJ AXB-12/PgnJ AXB-13/PgnJ AXB-15/PgnJ AXB-19/PgnJ AXB-19/PgnJ
AXB-2/PgnJ AXB-19/PgnJ AXB-23/PgnJ AXB-24/PgnJ AXB-4/PgnJ AXB-5/PgnJ AXB-6/PgnJ AXB-8/PgnJ BALB/cJ
BTBRT<+>tf/J BUB/BnJ BXA-1/PgnJ BXA-11/PgnJ BXA-12/PgnJ BXA-13/PgnJ BXA-14/PgnJ BXA-16/PgnJ BXA-2/PgnJ
BXA-24/PgnJ BXA-25/PgnJ BXA-26/PgnJ BXA-4/PgnJ BXA-7/PgnJ BXA-8/PgnJ BXD-1/TyJ BXD-11/TyJ BXD-12/TyJ
BXD-13/TyJ BXD-14/TyJ BXD-15/TyJ BXD-16/TyJ BXD-18/TyJ BXD-19/TyJ BXD-2/TyJ BXD-20/TyJ BXD-21/TyJ
BXD-24/TyJ BXD-27/TyJ BXD-28/TyJ BXD-29/TyJ BXD-31/TyJ BXD-32/TyJ BXD-33/TyJ BXD-34/TyJ BXD-36/TyJ
BXD-38/TyJ BXD-39/TyJ BXD-40/TyJ BXD-42/TyJ BXD-6/TyJ BXD-8/TyJ BXD-9/TyJ BXD-5/TyJ BXH-10/TyJ
BXH-14/TyJ BXH-19/TyJ BXH-2/TyJ BXH-4/TyJ BXH-6/TyJ BXH-7/TyJ BXH-8/TyJ BXH-9/TyJ C3H/HeJ C57BL/6J
C57L/J C58/J CBA/J CE/J DBA/2J FVB/NJ I/LnJ KK/HlJ LG/J LP/J MA/MyJ NOD/LtJ NON/LtJ NZB/BlNJ NZW/LacJ
PL/J RIIIS/J SEA/GnJ SM/J SWR/J
```

These genotypes, with the map, were used to reproduce two published HMDP GWAS (Farber et al. 2011,
bone mineral density; Bennett et al. 2010, HDL cholesterol) from their phenotypes:
[docs/replication.md](../docs/replication.md).

## The 1219-animal BED set (demo only: fake phenotypes)

| file | what |
|---|---|
| `snps.132k.clean.noX.bed/.bim/.fam` | PLINK 1 BED of 1219 inbred mice (one row per animal: replicate animals of inbred strains, so K has rank about 110) at 98,740 autosomal SNPs. SNP IDs are `mm37-<chr>-<bp>`, positions on build 37; every position is a row of the HMDP marker map. The `.bim` alleles are `1`/`2` (no letters). The `.fam` has numeric FID = IID `1`–`1219` and **no strain labels**. |
| `snps.132k.clean.noX.fake.phenos` | **FAKE phenotype**: `FID IID value`, one random value per animal, no header. It carries no biology; use it only to exercise the software (README example, tests). It cannot replicate or refute anything. |
| `snps.132k.clean.noX.pylmm.kin` | 1219 × 1219 kinship of all 98,740 SNPs from the original 2013 code, as text; the README example and the golden tests use it. `pylmm kinship --bfile data/snps.132k.clean.noX` recomputes it. |
| `snps.132k.clean.noX.pylmm.kin.id` | its ID sidecar (`FID IID` per row, in `.fam` order), added in Phase 5b.1 so `pylmm gwas` aligns the kinship by ID. |

## Checksums (SHA-256)

`tests/test_data_provenance.py` checks that every file tracked in git under `data/` is listed here and that every listed file exists and matches its hash (untracked local files are ignored; the tests skip when the data files are absent). From the
repository root:

```bash skip-ci: shasum is not part of the docs-test environment
cd data && shasum -a 256 -c <(grep -E '^[0-9a-f]{64}  ' README.md)
```

```text
70c9752cbc92898b259948f77e02d71d43f1710ad05e9550c2fb6e8ba6db812a  hmdp.liver.exprs
6898eb6beaa249f6d2504157c9e872ba85f14ce79a8c5e3bbe26da688e4edcb9  hmdp.liver.exprs.1
3c3e5eda7199fffc1f050251d8defab3556f1f81fa47c0d31417e71aa7a3c714  hmdp.liver.K.npdump
886a9ea7b7474ce425ac6e8d02b0c932f984804963c79fd65430f2739eb173e8  hmdp.liver.snps
74aeb38ddb6e3fe068842ad005edd0311e350a7920c4ff791150194f80c55e3b  hmdp.liver.snps.1000.npdump
488d48d8a316875cf2e5158cdc21082f35391e965885a961ec6b744a0769c140  hmdp.liver.snps.map.tsv
b37220798765fdd997828e9bbcfe49f34ee3524790b858fa91bbda230d1890fc  hmdp.liver.snps.npdump
c6bf102d7c7d2120c1619884a6fd9741a31d33d03a429d21ea42708c30c447ae  hmdp.liver.strains
dff11446d3e27affefda1181cc39aa74e424a2ad28ffe9ccd44c738c23a6f2af  snps.132k.clean.noX.bed
3a41c938206278e40381ea035b45d364e85a73084f3f61bdb23647c081637346  snps.132k.clean.noX.bim
5978e325f8123c5d851c3ee7cede8db267be9f6983d1f3ab90aae66735dbfe3b  snps.132k.clean.noX.fake.phenos
0631ebb0b29942f38dcdfc62bbf16060c2e10b2b07d306736635620be5bef209  snps.132k.clean.noX.fam
0dc73b0fcf3853d35e4b9b5768c78e6d4528675956c664e92a08cf4766041f1f  snps.132k.clean.noX.pylmm.kin
943dc30ca8550862aee1ac2218c20e3f2d8056184894f5dd78697b709f5c2982  snps.132k.clean.noX.pylmm.kin.id
```
