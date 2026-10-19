# I/O formats for pylmm: native support or conversion skill?

*Prepared 2026-10-18 for the planned I/O work (TODO.md Phase 6, especially 6.2 and 6.4). This is a read-only
review and no code was changed. Package versions, licenses and wheel tags were checked against the PyPI
JSON API on 2026-10-18.*

## 0. Bottom line

pylmm eigendecomposes an n × n kinship matrix, so its realistic working range is n of about 20–30k
samples (K alone is 3.2 GB in float64 at n = 20k). It is not a biobank-scale engine. That changes the
format question. pylmm does not need a high-throughput reader for every container. It needs:

1. **One fast, well-tested native genotype path (PLINK 1 BED)** that **keeps the .bim metadata** (chr, pos,
   ID, alleles). This is 6.2, and it is the prerequisite for LOCO (6.4) and for any standard output.
2. **A numpy array entry point.** An agent or user can then load anything (VCF, BGEN, Zarr, Hail) with
   any library and hand pylmm a dosage matrix plus sample and variant tables.
3. **ID-aware sample alignment** across genotype, phenotype, covariate and kinship files. Today
   alignment is partly by FID/IID and partly by row order. Row order for the kinship is the
   silent-failure risk flagged in Phase 4.
4. **A GWAS-SSF-compatible output table plus a JSON run-metadata sidecar.**
5. **Exactly one optional "rich" genotype reader: PGEN via `pgenlib`.** It brings dosages. BGEN and
   imputed VCF reach it with a lossless `plink2` conversion.

Everything else (VCF/BCF, BGEN, VCF Zarr, Hail) is a **skill-documented conversion** with plink2,
bcftools, vcztools or Hail. These conversions are one-liners, they are well established, and plink2
already handles their edge cases (multiallelics, half-calls, ID parsing, REF/ALT) better than a new
reader in pylmm would.

## 1. What pylmm does today (relevant facts from the code)

- `pylmm/input.py` `plink` class: it reads BED (SNP-major only), TPED/TFAM, and the "EMMA" text formats.
  `__next__` returns `(genotype_vector, bim_column_2)`. **chr, cM, pos, A1 and A2 are discarded.**
- BED decode is already a vectorized per-SNP numpy LUT (`_BED_CODE_VALUES`, `formatBinaryGenotypes`). The
  remaining 6.2 work is block decode (many SNPs per call, which feeds 6.1 batching) plus keeping the
  .bim columns.
- **Genotype coding and effect allele.** Code `00` maps to 0, `10` to 0.5 and `11` to 1.0, so the
  counted allele is **.bim column 6 (A2)**. When `plink2 --make-bed` writes the BED, A1 = ALT and A2 = REF
  (this is the documented convention; see vcztools `view-plink`, which is byte-identical to plink2). In
  that case pylmm's BETA is the effect of the **REF** allele. plink 1.9 without `--keep-allele-order` sets
  A1 = minor allele, so there pylmm's effect allele is the **major** allele. Neither is stated anywhere in
  the output.
- **The genotype scale is standardized.** `normGenotype=True` is the default, and `--removeMissingGenotypes`
  standardizes as well. **BETA and BETA_SD are therefore per SD of genotype, not per allele.** BETA_SD is
  the standard error of BETA, not an SD. T and p do not depend on the scale.
- **Phenotypes and covariates** use no header. They are matched on the (FID, IID) tuple against the
  .fam, and `-9`/`NA` mean missing. Individuals absent from the phenotype file are silently skipped
  (bug 10).
- **Kinship (`--kfile`)** is plain n × n text with **no IDs**. It is assumed to be in .fam order and is
  validated only for shape and finiteness (Phase 5.1, bug 16). The old FaST-LMM header style
  (`fastLMM_kinship_style`) did carry IDs but is not wired to the CLI.
- **Output**: a TSV with columns `SNP_ID BETA BETA_SD F_STAT P_VALUE`. Kinship output is `np.savetxt` of K,
  plus `.kva`/`.kve` text (bug 7: `--eigen` cannot read them back).
- `requires-python >= 3.10`, and the dependencies are numpy and scipy only.

## 2. Genotype inputs

### 2.1 Ecosystem facts (verified 2026-10-18)

| Library | Latest (date) | License | Wheels, cp310–cp314 | macOS arm64 / Linux | Notes |
|---|---|---|---|---|---|
| **bed-reader** | 1.1.0 (2026-06) | Apache-2.0 | 310–314 | yes / yes (also Windows, mac x86) | Rust core from the FaST-LMM group. Has `count_A1`, float32/int8 output, sample/variant slicing, cloud URLs. Deps: numpy only |
| pandas-plink | 2.3.2 (2025-05) | MIT | 310–312 only | yes / yes | Pins `numpy<2`, `pandas<3`; pulls in dask, pandera, xarray. Last commit 2025-05. **Avoid** |
| pysnptools | 0.5.15 (2026-08) | Apache-2.0 | pure py | n/a | Depends on bed-reader plus pandas, h5py, psutil, cloudpickle. Too heavy for our needs |
| **Pgenlib** (pgenlib) | 0.95.1 (2026-10) | **LGPL-3.0** | 310–314 | yes / yes (no mac x86) | Official plink2 reader/writer (Chang). Reads hard calls, **dosages**, phase, multiallelic. Deps: numpy only |
| bgen-reader | 4.0.9 (2024-10) | MIT | pure py, needs cbgen | via cbgen | Pins `dask<2025`, `xarray<2025`. Stale |
| cbgen | 1.0.6 (2024-06) | MIT | 39–312 only | yes / yes | No 3.13/3.14 wheels. Last commit 2024-06. **Stale** |
| **bgen** (jeremymcrae) | 1.10.3 (2026-09) | MIT | 39–315 | yes / yes | Small, maintained BGEN reader/writer. Best BGEN option if one is ever needed |
| cyvcf2 | 0.34.0 (2026-06) | MIT | 39–314 | yes / yes | htslib wrapper. Fast for a VCF parser, but still a text parse |
| pysam | 0.24.1 (2026-09) | MIT | 39–315 | yes / yes | General htslib binding. Heavier |
| scikit-allel | 1.3.13 (2024-09) | MIT | 310–313 | yes / yes | In maintenance mode (its successor is sgkit). No 3.14 wheels |
| sgkit | 0.10.0 (2025-04) | Apache-2.0 | pure py | n/a | Active repo (commits Oct 2026), but the release pins `numpy<2.2`, `dask<=2024.8`, `zarr<3`. Heavy dependency |
| bio2zarr | 0.2.1 (2026-06) | Apache-2.0 | pure py, py>=3.11 | n/a | `vcf2zarr`, `plink2zarr`, `tskit2zarr` |
| vcztools | 0.2.0 (2026-06) | Apache-2.0 | 311–313 | yes / yes | bcftools-like CLI for VCF Zarr, including `view-plink` (VCZ to BED) and `view-bgen` |
| hail | 0.2.139 (2026-08) | MIT | pure py, needs JVM/Spark | n/a | Out of scope as a dependency |

Sources: PyPI JSON API for each package (https://pypi.org/pypi/&lt;name&gt;/json);
https://github.com/fastlmm/bed-reader; https://github.com/chrchang/plink-ng/tree/master/2.0/Python;
https://github.com/jeremymcrae/bgen; https://github.com/limix/cbgen; https://github.com/limix/pandas-plink;
https://github.com/brentp/cyvcf2; https://github.com/sgkit-dev/sgkit; https://github.com/sgkit-dev/bio2zarr;
https://github.com/sgkit-dev/vcztools.

**Python floor.** Python 3.10 reaches end of life in October 2026 (PEP 619, https://peps.python.org/pep-0619/).
pandas 3.x and pyarrow ≥ 26 require Python ≥ 3.11, and the last cp310 wheels are pandas 2.3.3 and
pyarrow 25.0.1. Any pandas or Parquet extra therefore argues for moving `requires-python` to ≥ 3.11 at
cutover. That decision is Nick's.

### 2.2 Input formats accepted by current GWAS tools

| Tool | BED | PGEN | BGEN | VCF/BCF | Dosage text | Notes and source |
|---|---|---|---|---|---|---|
| GEMMA 0.98 | yes | no | no | no | BIMBAM mean-genotype (dosage) | https://github.com/genetics-statistics/GEMMA |
| GCTA (MLMA, fastGWA) | yes (`--bfile`, `--mbfile`) | yes (`--pfile`) | yes (`--bgen`, `--mbgen`) | no | legacy MACH | https://yanglab.westlake.edu.cn/software/gcta/#fastGWA |
| BOLT-LMM 2.4/2.5 | yes (model fitting needs BED) | no | yes (test only) | no | IMPUTE2/dosage | https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html |
| REGENIE 4.x | yes | yes | yes (the main path for imputed data) | no | no | https://rgcgithub.github.io/regenie/options/ |
| SAIGE (step 2) | yes | no | yes (.bgi) | yes (`--vcfField=DS/GT`, .csi) | SAV | https://saigegit.github.io/SAIGE-doc/docs/single_step2.html |
| plink2 `--glm` | yes | native | via import | via import | via import | https://www.cog-genomics.org/plink/2.0/input |
| sgkit | via `read_plink` | via bio2zarr/VCZ | via reader | via vcf2zarr | n/a | https://sgkit-dev.github.io/sgkit/ |

**Pattern:** every mixed-model tool reads BED, and the tools that handle imputed data add **BGEN and/or
PGEN** for dosages. None of the LMM tools reads raw VCF for the heavy step except SAIGE step 2.

### 2.3 Per-format assessment

**PLINK 1 BED/BIM/FAM.** This is the universal GRM and LMM input, with 2 bits per genotype and hard calls
only. A file with n = 10k and m = 500k is 1.25 GB on disk. Dense float64 would be 40 GB, so the reader
must stream variant blocks; float32 halves that. The numpy LUT decode is already in core. bed-reader is
multithreaded Rust and supports float32 and slicing, which makes it a good optional speed-up
(`pylmm[fast]`, Apache-2.0, numpy-only dependency). To match the goldens, pass `count_A1=False`, because
pylmm counts A2. Lost relative to the source data: dosages, multiallelics, phase, and anything beyond
biallelic A1/A2. No genome build is recorded. **Native (core).**

**PLINK 2 PGEN/PVAR/PSAM.** This is plink2's native format and is far more compact for sequence data.
It stores dosages (precision 1/16384), multiallelics and phase. PVAR carries REF/ALT explicitly and the
PSAM header carries IDs (`#IID` or `#FID IID`). pgenlib is official, maintained and has wheels for
cp310–314 on macOS arm64 and Linux. It is **LGPL-3.0**, which is fine as an optional, dynamically
imported extra but should not be a core dependency of an MIT package. Conversion to BED is a one-liner,
but it **loses dosages** (hard-call threshold 0.1 by default; dosages of exactly 0.5/1.5 become missing)
and errors on multiallelics. PGEN is the one format whose native support buys something a skill
cannot: imputed dosages without loss. **Optional extra `pylmm[pgen]`.**

**BGEN (UK Biobank, imputed).** It holds genotype probabilities and is often zstd-compressed. `plink2
--bgen ... --make-pgen` is lossless up to PGEN dosage precision, which is plenty for an additive model.
Use `ref-first` for UK Biobank. The Python BGEN ecosystem is split. cbgen and bgen-reader are stale and
lack 3.13/3.14 wheels. `bgen` (jeremymcrae) is healthy but would be a second dosage reader to maintain.
**Skill: convert to PGEN (or to BED for hard calls), then use the pgen extra.** Revisit a native `bgen`
extra only if users hit the conversion often.

**VCF/BCF.** This is the interchange format, but it is slow to parse (text, per-sample FORMAT fields)
and its edge cases are hard: multiallelics, half-calls, ID formats, GT versus DS. plink2 and bcftools
already handle them. No LMM tool in the comparison table depends on raw-VCF throughput except SAIGE
step 2. **Skill: `bcftools norm -m -any` then `plink2 --vcf/--bcf ... --make-bed` (hard calls) or `--make-pgen`
with `dosage=DS` (imputed).**

**VCF Zarr (VCZ) / sgkit.** This is an analysis-ready, chunked, columnar encoding of VCF. The spec and
tools are published (Czech et al., *GigaScience* 2025, https://doi.org/10.1093/gigascience/giaf049;
spec https://github.com/sgkit-dev/vcf-zarr-spec). It is agent-friendly, since it opens as an xarray
Dataset, and it is growing in large cohorts (Genomics England, Our Future Health, All of Us case
studies). However, the Python stack is heavy (zarr ≥ 3 needs py ≥ 3.12, plus xarray and dask), and
sgkit's release pins old numpy and dask. `vcztools view-plink` writes BED that is byte-identical to
plink2's. **Skill now (`vcztools view-plink`), or the numpy-array entry point
(`ds.call_genotype.sum(-1)` gives dosages to pass straight in). Revisit later as an optional extra.**

**Hail MatrixTable.** It needs a JVM/Spark and is large-cohort infrastructure. Users export with
`hl.export_plink` (after `hl.split_multi_hts`) (https://hail.is/docs/0.2/methods/impex.html). **Skip;
document the export in a skill.**

**TPED/TFAM and EMMA text.** These are legacy formats with golden coverage today. **Keep them, frozen**:
no new features, and they should still go through the new alignment layer. Consider deprecating EMMA
at the next breaking window. They are useful only for the original pylmm examples.

**Dosage versus hard calls.** The LMM math in pylmm already works on real-valued genotypes after
standardization, so dosages need no algorithmic change. The reader contract should be **"float dosage
of the counted allele in [0, 2], NaN = missing"**. BED yields {0, 1, 2, NaN}, while PGEN and arrays yield
real values. TPED/EMMA currently produce {0, 0.5, 1}, which is harmless after standardization but must
be fixed for per-allele effect sizes (see §4.3).

### 2.4 Genotype summary

| Format | Recommendation | Rationale |
|---|---|---|
| PLINK 1 BED/BIM/FAM | **Native (core)**, numpy LUT block decoder; bed-reader as `pylmm[fast]` | Universal LMM input, simple format, already supported; 6.2 |
| numpy array plus sample/variant tables | **Native (core)** | Lets any library feed pylmm (agents, sgkit, Hail, cyvcf2); zero dependencies |
| PLINK 2 PGEN | **Optional extra `pylmm[pgen]`** (pgenlib, LGPL) | Only lossless route for imputed dosages; official, maintained reader with cp310–314 wheels |
| BGEN | **Skill** (to PGEN via plink2); later possibly `bgen` extra | Lossless conversion to PGEN; Python BGEN libraries are split or stale |
| VCF/BCF | **Skill** (bcftools norm, then plink2) | Slow to parse, many edge cases plink2 already handles |
| VCF Zarr / sgkit | **Skill** (`vcztools view-plink`) or array API; revisit | Heavy stack; array API covers ad hoc use |
| Hail MatrixTable | **Skip** (skill: `hl.export_plink`) | JVM; out of scope |
| TPED/TFAM, EMMA | **Keep, frozen** (consider deprecating EMMA) | Legacy; golden-tested |

## 3. Phenotype, covariate and kinship inputs

### 3.1 Phenotype and covariate files

These are the conventions to accept (plink2: https://www.cog-genomics.org/plink/2.0/input#pheno;
REGENIE: https://rgcgithub.github.io/regenie/options/):

- An optional header line that starts with `FID`, `#FID`, `IID` or `#IID`. Leading columns are FID+IID
  or IID alone. Other header columns are trait or covariate names.
- With no header, the file is read in the legacy pylmm/PLINK 1 style (FID IID v1 v2 ...), and the
  columns are named `PHENO1`... or `COVAR1`....
- Missing values are `NA`, `nan` (any case) and `-9` (PLINK default; it must be possible to disable
  `-9` for quantitative traits where −9 is a real value). REGENIE accepts `NA` only.
- Whitespace-delimited (tab or space). Add **CSV/TSV with header** (sniffed by extension or by `--sep`).
  Parquet can be read only through the optional pandas/pyarrow extra.
- Select by name: `--pheno-name BMI`, `--covar-name age,sex,PC1-PC10` (plink2 style). Integer
  `--pheno` indices stay for legacy use only.
- Categorical covariates: refuse non-numeric values with a clear error. Skip one-hot encoding; agents
  can do it in pandas.

### 3.2 Kinship and GRM interop

| Format | What it is | Recommendation |
|---|---|---|
| pylmm text `.kin` (n × n, whitespace, optional `.gz`) | Current input and output; no IDs | **Native**; add an ID sidecar `.kin.id` (FID IID, .fam style) on write, and use it on read when present |
| GEMMA `-gk 1/2` output `*.cXX.txt` / `*.sXX.txt` | n × n tab text, .fam order, no IDs (https://github.com/genetics-statistics/GEMMA) | **Already readable** as plain text. Order must match the .fam; document it |
| GCTA GRM binary `.grm.bin` / `.grm.N.bin` / `.grm.id` | float32 lower triangle including the diagonal, row-major; `.N.bin` holds per-pair SNP counts (float32); `.id` holds FID IID with no header (https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM) | **Native read and write** (about 40 lines of numpy, no dependencies). It carries IDs, is written by `gcta --make-grm` and `plink2 --make-grm-bin`, and is read by GCTA, fastGWA (sparse variant aside) and MTG2. Makes 6.4's "GCTA-style GRM" validation simple |
| plink2 `--make-rel` `.rel` / `.rel.bin` / `.rel.id` | square or triangle, text or binary (f64, or f32 with `bin4`), with IDs (https://www.cog-genomics.org/plink/2.0/distance#make_rel) | **Native read** of square and `bin`/`bin4`. Small cost, with IDs |
| FaST-LMM header style (IDs in the first row and column) | Legacy reader code exists but is not wired up | Drop it or route it through the alignment layer |
| eigendecomposition `.kva`/`.kve` text | bug 7 | Replace with a single `.npz` (`eigvals`, `eigvecs`, `iid`, `fid`, `pylmm_version`, `kinship_sha256`) so `--eigen` cannot be paired with the wrong K |

### 3.3 IDs as first-class: the alignment layer

Today phenotypes and covariates are matched by (FID, IID) with silent drops, and K is matched by row
order. The proposal:

- Every input produces a **sample key array**. Genotype keys come from .fam/.psam, phenotype and
  covariate keys from file columns, and kinship keys from `.grm.id`/`.rel.id`/`.kin.id`.
- `--id-match {iid,fid-iid}` sets the matching mode. The default is **`iid`**, because plink2 and VCF
  imports often write FID = 0 or FID = IID inconsistently, while REGENIE and plink2 both treat IID as
  primary. Use `fid-iid` for legacy family data.
- **Analysis set** = intersection of all inputs, minus samples with a missing phenotype or covariate,
  **in genotype file order**. Every input is reindexed to that order (K is subset with `K[np.ix_(i, i)]`).
- **Errors, not warnings**, in these cases: duplicate keys in any input; an empty intersection; a K
  with no IDs whose dimension is not exactly the genotype n.
- **Loud warning** when a K has no IDs and is assumed to be in genotype order. This is the Phase 4
  silent-failure case. Recommend that pylmm-written kinships always carry IDs, so the warning only
  fires for foreign files.
- Record counts in the metadata JSON: n per input, n intersected, and n dropped per reason.
- Where it fits: this is a robustness fix more than an improvement, and it changes CLI behavior. It
  fits the **Phase 5 input-validation bundle (bugs 10, 11, 14) and the 5b breaking-change window**
  better than Phase 6.

### 3.4 Phenotype, covariate and kinship summary

| Format | Recommendation | Rationale |
|---|---|---|
| PLINK-style pheno/covar, with or without header (FID/IID or IID; NA/nan/-9) | **Native** | Lingua franca; plink2/REGENIE-compatible |
| CSV/TSV with header | **Native** (stdlib `csv` or numpy) | Agent-friendly; trivial |
| Parquet pheno/covar | **Optional extra `pylmm[parquet]`** | Useful for agents; pyarrow is large and needs py ≥ 3.11 |
| pylmm text kinship plus `.kin.id` | **Native** | Existing; add IDs |
| GCTA GRM binary | **Native read and write** | IDs, interop with GCTA/plink2; tiny code |
| plink2 `.rel(.bin)` | **Native read** | IDs; tiny code |
| GEMMA `cXX.txt` | **Native (already)**; document | Same layout as pylmm text |
| Eigendecomposition | **Native `.npz` with IDs** (replaces `.kva`/`.kve`) | Fixes bug 7; ties eigenvectors to IDs |

## 4. Outputs

### 4.1 The de facto summary-statistics conventions

**GWAS-SSF (GWAS Catalog standard, v1.1.0).** https://www.ebi.ac.uk/gwas/docs/summary-statistics-format and
https://github.com/EBISPOT/gwas-summary-statistics-standard
- The mandatory columns must appear **in this order**: `chromosome` (1–22, 23 = X, 24 = Y, 25 = MT),
  `base_pair_location` (integer > 0), `effect_allele` and `other_allele` (sequences of ACGT), one of
  `beta`/`odds_ratio`/`hazard_ratio`/`z_score`, `standard_error`, `effect_allele_frequency`, and one of
  `p_value`/`neg_log_10_p_value`.
- Optional columns: `variant_id` (chr_pos_ref_alt), `rs_id`, `info`, `ci_lower`, `ci_upper`, `ref_allele`
  (EA/OA), `n`. Any extra columns are allowed.
- Missing values are `#NA`. The format is TSV with a header, one file per GWAS, and may be compressed.
  A `<file>-meta.yaml` sidecar carries the genome assembly, coordinate system, md5 and sample/trait
  details.
- Effect allele: "where possible", effect allele = ALT and other allele = REF.
- Validator: `gwas-sumstats-tools` 2.0.2 (https://github.com/EBISPOT/gwas-sumstats-tools), which provides
  the `gwas-ssf validate|format|read|gen-meta` commands. It **requires Python ≥ 3.13** and pandas/pandera.
  `--min-rows` defaults to 100,000, so pass `-m 1` for small test runs; `-z` allows p = 0. Run it from a
  skill with `uvx`; it must not be a dependency.

**plink2 `--glm` (`.glm.linear`)** (https://www.cog-genomics.org/plink/2.0/formats#glm_linear). The default
columns are `#CHROM POS ID REF ALT PROVISIONAL_REF? A1 OMITTED A1_FREQ TEST OBS_CT BETA SE T_STAT P
ERRCODE`. A1 is the counted/effect allele (ALT by default). `NEG_LOG10_P` is optional (via `cols=`).

**Mixed-model tool outputs:**

| Tool | Columns (abridged) | Effect allele |
|---|---|---|
| REGENIE step 2 | `CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA` | ALLELE1 (ALT) |
| SAIGE step 2 | `CHR POS MarkerID Allele1 Allele2 AC_Allele2 AF_Allele2 MissingRate BETA SE Tstat var p.value` | **Allele2** |
| BOLT-LMM | `SNP CHR BP GENPOS ALLELE1 ALLELE0 A1FREQ F_MISS BETA SE P_BOLT_LMM_INF P_BOLT_LMM` | ALLELE1 |
| GCTA fastGWA | `CHR SNP POS A1 A2 N AF1 BETA SE P` | A1 |
| GCTA `--mlma` | `Chr SNP bp A1 A2 Freq b se p` | A1 |
| GEMMA `-lmm` | `chr rs ps n_miss allele1 allele0 af beta se logl_H1 l_remle p_wald` | allele1 |

Sources: https://rgcgithub.github.io/regenie/options/#output;
https://saigegit.github.io/SAIGE-doc/docs/single_step2.html;
https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html;
https://yanglab.westlake.edu.cn/software/gcta/#fastGWA; https://github.com/genetics-statistics/GEMMA.

**Downstream consumers:**
- LDSC `munge_sumstats.py` needs SNP (rsID), A1 (= effect allele), A2, a signed statistic
  (Z/BETA/OR/LOG_ODDS), P and N, and merges on HapMap3 rsIDs
  (https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format). **rsIDs matter** for LDSC.
- MungeSumstats (Bioconductor) auto-detects most header spellings, uses **A2 = effect allele** internally
  (IEU GWAS-VCF style), flips against the reference genome, and can emit LDSC or GWAS-VCF format
  (https://bioconductor.org/packages/MungeSumstats/). Allele direction conventions genuinely conflict
  across tools, so **explicit column names (`effect_allele`, `other_allele`) beat A1/A2**.

**What all of these share:** CHR, POS, ID, two named alleles with an explicit effect allele, the effect
allele frequency, N, a per-allele BETA, SE and P. pylmm emits none of CHR, POS, alleles, frequency or N.
Its `BETA_SD` is the SE but is named like a standard deviation, and its BETA is on the standardized
scale.

### 4.2 Proposed default output schema

The file is `<out>.tsv.gz`, tab-separated with a header. Missing values are written as `NA` (pandas,
R and LDSC read it as missing). A `--ssf` switch writes `#NA` and numeric chromosome codes, and emits a
`-meta.yaml`, so the validator passes. Mandatory columns come first, in GWAS-SSF order:

```
chromosome  base_pair_location  effect_allele  other_allele  beta  standard_error  effect_allele_frequency  p_value \
  variant_id  rs_id  n  t_stat  neg_log_10_p_value  missing_rate
```

| Column | Source / definition |
|---|---|
| chromosome | .bim/.pvar chr as written. Mapped X/Y/MT → 23/24/25 only under `--ssf` (keeps LOCO grouping simple) |
| base_pair_location | .bim col 4 / .pvar POS (1-based) |
| effect_allele | **The counted allele.** Proposal: count **A1 (= ALT for plink2-written files)**, which matches plink2 `--glm`, REGENIE, fastGWA, GEMMA and GWAS-SSF's "effect = ALT". This flips the sign relative to today's A2 coding. Only the new schema does this; the legacy writer keeps A2 |
| other_allele | the other .bim allele |
| beta | **per-allele** effect: β_std / sd(dosage₀₋₂), with sd taken over the non-missing samples, as `normalizeGenotype` does today |
| standard_error | same rescaling of the current `BETA_SD` |
| effect_allele_frequency | mean dosage / 2 over non-missing samples in the analysis set (already computed in `normalizeGenotype`) |
| p_value | as today; double precision |
| variant_id | the .bim/.pvar ID verbatim (pylmm's current `SNP_ID`). Note that GWAS-SSF wants `chr_pos_ref_alt`; `--ssf` builds it and moves the original ID to `rs_id` when it matches `^rs\d+$` |
| rs_id | ID if it matches `^rs\d+$`, else NA |
| n | samples in the analysis set (with mean imputation, this is the same for every SNP) |
| t_stat | signed Wald t. F_STAT = t² is dropped in the new schema |
| neg_log_10_p_value | from `logsf`, survives p < 1e-308. Needs the 6.3 log-p fix (bug 5) |
| missing_rate | fraction of genotypes that were missing before imputation, or excluded under `--removeMissingGenotypes` |

**Schema notes:**
- **Legacy output** (`SNP_ID BETA BETA_SD F_STAT P_VALUE`) stays available behind `--output-format
  legacy` so the goldens stay green. Per Nick's Phase 6 rule, the default switch needs a decision,
  because the numbers change: sign (allele) and scale (per allele).
- The beta scaling and allele change belong to the writer, not the solver. The LMM still sees
  standardized genotypes, so t and p are unchanged; only the reported beta and SE are rescaled.
- **TSV.gz versus Parquet.** TSV.gz is the default: universal, R/LDSC/MungeSumstats/GWAS-Catalog ready,
  and stdlib `gzip` only. bgzip/tabix-compatible output needs htslib, so a skill documents
  `bgzip`+`tabix` instead. **Parquet is optional** (`--out-format parquet`, needs `pylmm[parquet]`). It is
  typed, 5–10× faster to load for agents, and carries the run metadata in the schema's key-value
  metadata. Write it as a single file sorted by chromosome and position.

### 4.3 JSON run-metadata sidecar (`<out>.json`)

This is always written: it is cheap, and it is the single most useful thing for agents and for
reproducibility.

```json
{
  "schema": "pylmm.run/1",
  "pylmm_version": "1.0.0", "git_sha": null,
  "command": ["pylmm", "gwas", "--bfile", "..."],
  "started_at": "2026-10-18T12:00:00Z", "finished_at": "...",
  "environment": {"python": "3.13.1", "numpy": "2.3.5", "scipy": "1.16.0", "blas": "openblas 0.3.28", "platform": "macOS-15-arm64"},
  "inputs": {
    "genotypes": {"path": "geno", "format": "bed", "n_samples": 1000, "n_variants": 500000, "counted_allele": "A1"},
    "phenotype": {"path": "pheno.txt", "column": "BMI"},
    "covariates": {"path": "cov.txt", "columns": ["age", "sex", "PC1"], "intercept": true},
    "kinship": {"path": "geno.grm.bin", "format": "gcta-grm", "has_ids": true, "sha256": "..."}
  },
  "samples": {"id_match": "iid", "n_genotyped": 1000, "n_pheno": 990, "n_covar": 1000, "n_kinship": 1000,
              "n_analysed": 985, "dropped": {"missing_pheno": 10, "missing_covar": 5, "not_in_kinship": 0}},
  "model": {"estimator": "REML", "test": "wald", "refit": false, "loco": false, "genotype_scale": "standardized",
            "missing_genotypes": "mean-impute"},
  "null_fit": {"h2": 0.42, "sigma_g2": 0.39, "sigma_e2": 0.54, "delta": 1.38, "log_likelihood": -1234.5,
               "h2_at_bound": false, "eig_clamped": 3},
  "results": {"path": "out.tsv.gz", "n_rows": 500000, "n_na": 12, "lambda_gc": 1.01, "min_p": 3e-12,
              "columns": ["chromosome", "..."]},
  "timings_s": {"read_genotypes": 1.2, "kinship": 0.0, "eigh": 3.4, "null_fit": 0.2, "scan": 40.1, "total": 45.0},
  "warnings": ["kinship had no IDs; assumed genotype order"]
}
```

Notes on the sidecar:
- `h2_at_bound` surfaces the 0.99 cap (bugs 3 and 4).
- `lambda_gc` is cheap to compute and is the first thing an agent checks.
- `warnings` turns the silent failures from Phase 4 into something an agent can read.
- `pylmm kinship` writes the same sidecar: n, m, SNPs used, normalization (bug 8), and IDs file.

### 4.4 Kinship outputs

- The default is `.kin` text (backward compatible) **plus `.kin.id`** (FID IID, no header).
- `--grm-format gcta` writes `.grm.bin`, `.grm.N.bin` and `.grm.id`. It is a 4-byte float lower
  triangle, so the file is half the size, and it is readable by GCTA, plink2 and other tools. This helps
  6.4 validation against GCTA.
- The eigendecomposition goes to a single `.npz` with IDs (§3.2).

### 4.5 Output summary

| Output | Recommendation | Rationale |
|---|---|---|
| GWAS-SSF-ordered TSV.gz (schema §4.2) | **Native, new default** (legacy behind a flag) | Readable by LDSC, MungeSumstats, the GWAS Catalog, pandas and R; carries chr/pos/alleles |
| `--ssf` strict mode plus `-meta.yaml` | **Native (small)** | Catalog submission; validator passes |
| Parquet | **Optional extra** | Fast and typed for agents; heavy dependency |
| JSON run sidecar | **Native, always** | Reproducibility, agent introspection, surfaces warnings |
| Legacy 5-column text | **Native, frozen** behind `--output-format legacy` | Goldens and migration |
| GCTA GRM binary | **Native write** (opt-in) | Interop; IDs |
| `.kin` + `.kin.id` | **Native default** | Backward compatible, plus IDs |
| GWAS-VCF (IEU), LDSC `.sumstats` | **Skill** (MungeSumstats or `munge_sumstats.py`) | Downstream tools already do it |
| bgzip + tabix-indexed output | **Skill** | Needs htslib |

## 5. The agent angle

The ad hoc workflows agents actually run are: load the results, filter `p < 5e-8`, join with gene
annotations, draw a QQ/Manhattan plot, compute λ, compare against a Catalog study, and rerun with
different covariates. What helps most:

1. **Self-describing outputs**: named, typed columns with an explicit effect allele, plus the JSON
   sidecar. `pd.read_csv("out.tsv.gz", sep="\t")` should just work, and so should `json.load`.
2. **One library call that mirrors the CLI**, returning a result object rather than writing files:
   ```python
   res = pylmm.gwas(genotypes="geno", phenotype="pheno.txt", pheno_name="BMI",
                    covariates="cov.txt", kinship="geno.grm")   # or arrays + id tables
   res.null_fit.h2; res.samples.dropped; res.table            # dict of 1-D numpy arrays
   df = res.to_pandas()                                         # lazy import; ImportError names the extra
   res.write("out")                                             # tsv.gz + json
   ```
   The array entry point, `pylmm.gwas(G=ndarray, sample_ids=..., variants={"chrom":..., "pos":...}, y=..., K=...)`,
   is what makes VCF Zarr, Hail and BGEN reachable by agents without new readers in pylmm.
3. **Clear, typed errors on alignment problems**, such as duplicate IIDs, no overlap, or K without IDs
   whose size does not match. Agents recover from errors; they cannot recover from silent row shifts.

**pandas versus a plain numpy API.**
- Core stays **numpy + scipy**: readable, light, and it keeps the cp310 floor possible.
- Results are a small frozen dataclass of 1-D numpy arrays (a column store). Variant metadata such as
  chrom and ID is held as numpy string/object arrays.
- `to_pandas()`, `to_polars()` and Parquet I/O import lazily via extras. This mirrors bed-reader's
  approach (numpy-only core, pandas in extras).
- **Skip xarray as a dependency.** A `to_xarray()` with sgkit-style dimension names (`variants`,
  `samples`) is a cheap later add if the sgkit interop story grows.
- **pandas would be justified as a core dependency only if** the phenotype/covariate reader needs its
  CSV sniffing and dtype handling. It does not: whitespace and CSV files with a header plus numeric
  columns are about 60 lines with the stdlib `csv` module or `np.genfromtxt`.

**Proposed pip extras:** `pylmm[fast]` = bed-reader; `pylmm[pgen]` = Pgenlib; `pylmm[pandas]` = pandas;
`pylmm[parquet]` = pyarrow (+ pandas); `pylmm[all]`.

## 6. Proposed I/O architecture

```
pylmm/io/
  samples.py     SampleTable(fid: ndarray[str] | None, iid: ndarray[str]); align(*tables, mode) -> AlignResult
  variants.py    VariantTable(chrom, pos, id, counted_allele, other_allele, cm)  # from .bim/.pvar/arrays
  genotypes.py   GenotypeSource protocol + BedSource (numpy LUT | bed-reader) + ArraySource
                 + PgenSource (lazy `import pgenlib`) + legacy TpedSource/EmmaSource
  tables.py      read_table(path, kind="pheno"|"covar", names=..., missing=("NA","nan","-9")) -> (SampleTable, names, ndarray)
  kinship.py     read_kinship(path) -> (K, SampleTable | None); write_text/.kin.id, read/write_gcta_grm, read_plink2_rel, eig npz
  sumstats.py    write_sumstats(result, path, fmt="tsv.gz"|"parquet"|"legacy", ssf=False); write_metadata(json)
```

```python
class GenotypeSource(Protocol):
    samples: SampleTable
    variants: VariantTable
    def read_block(self, variant_idx: slice | ndarray, sample_idx: ndarray | None = None,
                   dtype=np.float32) -> ndarray:      # (n_samples, n_block), counted-allele dosage 0..2, NaN missing
    def iter_blocks(self, block_size: int = 4096, variant_idx=None, sample_idx=None) -> Iterator[tuple[slice, ndarray]]
```

**Design rules:**
- Readers **only decode**. Standardization and mean imputation move to a single
  `standardize_block(G) -> (Z, mean, sd, n_obs)` in the core. Today `normalizeGenotype` sits inside the
  reader. Moving it out is what lets the writer report allele frequency and the per-allele beta, and
  lets 6.1 batch the work.
- `variant_idx` selection by chromosome is exactly what **6.4 LOCO** needs: build K₋c from
  `iter_blocks(variant_idx=chrom != c)` and test `chrom == c`.
- `sample_idx` comes from `align()`, so readers never see pheno/covar/K files.
- **The legacy `input.plink` class stays as a thin shim**, or is dropped under the 5b "no aliases"
  rule with a migration-skill entry, so the goldens keep running.

## 7. Ordered implementation plan

| # | Item | Effort | Phase fit / dependencies |
|---|---|---|---|
| 0 | **Conversion skills** (§8) in `.claude/skills/` plus AGENTS.md "supported inputs" table | S | **5c** now; no code. Unblocks VCF/BGEN/Zarr users immediately |
| 1 | **VariantTable from .bim** (chr, pos, id, A1, A2, cM) + **block BED decode** (numpy LUT over a (b × bytes) array) + `GenotypeSource` protocol + `ArraySource`; bed-reader as `[fast]` with `count_A1` mapped explicitly. Add the decoder tail-length tests from the independent review (n mod 4 ≠ 0) | S–M | **This is 6.2**, widened slightly. Prerequisite for 3, 6 and **6.4 LOCO**; pairs with **6.1** batching (block in, block out). Identical arrays mean goldens stay green |
| 2 | **ID-aware alignment layer** + header-aware pheno/covar reader (named columns, NA/nan/-9, CSV/TSV) + **kinship IDs** (`.kin.id` write and read; error/warn rules) | M | Fold into the **Phase 5 input-validation bundle** (bugs 10, 11, 14) and the **5b** CLI flags (`--pheno-name`, `--covar-name`, `--id-match`). Independent of 6.x |
| 3 | **New sumstats writer** (§4.2 schema, TSV.gz) + **JSON sidecar** + `--output-format legacy`; move standardization out of the reader to get AF/n/missing_rate and the per-allele beta | S–M | Needs 1 (chr/pos/alleles). The schema can be designed in 5b (breaking window) and lands with or right after 6.1/6.2. `neg_log_10_p_value` needs **6.3** (bug 5). Default-on is Nick's call (the sign and scale of beta change) |
| 4 | **GCTA GRM bin read/write**, plink2 `.rel` read, eigendecomposition `.npz` with IDs (fixes bug 7) | S | Uses 2's SampleTable. Directly supports **6.4** (validate the GCTA-style GRM against `gcta --make-grm`) |
| 5 | **Library result object** (`pylmm.gwas(...)` returning arrays; `to_pandas`, `to_parquet`, `write`) + extras `[pandas]`, `[parquet]` | S | After 3. Pairs with the 5c library guide; the docs snippets run in CI |
| 6 | **PGEN reader extra** (`PgenSource` via pgenlib: dosages, REF/ALT from .pvar, IDs from .psam; reject multiallelics with a clear "split with plink2" message) | M | After 1 and 2. Benefits from 6.1 batching. Validate dosage results against `plink2 --glm` on a linear model with K = I, and against REGENIE/GCTA on a small imputed set |
| 7 | **`--ssf` strict mode** + `-meta.yaml` generator + CI check with `uvx gwas-ssf validate -m 1` | S | After 3. Optional; useful for Phase R replication against Catalog studies |
| 8 | Deferred: native BGEN (`bgen` package) extra; VCF Zarr reader / `to_xarray`; Parquet genotype I/O | M each | Only on demand. The skills plus PGEN cover these |

**Dependency notes for Phase 6:**
- **6.4 LOCO needs** item 1 (per-variant chromosome plus chromosome-filtered block iteration) and
  item 3 (chromosome in the output). A LOCO K built per chromosome can be written as GCTA GRMs (item 4)
  for inspection.
- **6.1 batching** should consume `iter_blocks` from item 1, so design the block API once.
- **6.3** (log-p) feeds `neg_log_10_p_value`. Until then, compute −log10 p from p and accept underflow
  (as NA).

## 8. Conversion recipes for skills

These recipes need plink2 ≥ 2.0 (alpha 6+) and bcftools ≥ 1.17. Each skill should end with "then run
`pylmm gwas --bfile geno ...`" and should state which allele pylmm will report as the effect allele.

**8.1 VCF/BCF (called genotypes) to BED.**
```bash
# split multiallelics, left-normalize (needs the matching reference FASTA), keep SNVs/indels
bcftools norm -m -any -f ref.fa -Ob -o split.bcf in.vcf.gz && bcftools index split.bcf
plink2 --bcf split.bcf \
  --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 100 missing \
  --vcf-half-call missing --max-alleles 2 \
  --double-id \
  --autosome --snps-only just-acgt --maf 0.01 --geno 0.05 \
  --make-bed --out geno
```
Caveats:
- `--make-bed` fails on multiallelic variants: the PLINK 1 .bim cannot represent them. Split first, or
  keep `--max-alleles 2`.
- The default `--vcf-half-call` mode is `error`.
- Choose the ID handling: `--double-id` sets FID = IID; the default leaves FID unset/0; `--id-delim`
  parses `FID_IID`. With pylmm's planned `--id-match iid` this matters less; today pylmm matches the
  phenotype file on (FID, IID), so the phenotype file must use the same FID.
- plink2 writes A1 = ALT and A2 = REF.
- Phase is dropped. Single quotes keep bash from expanding `$r`/`$a`.

Sources: https://www.cog-genomics.org/plink/2.0/input#vcf, https://www.cog-genomics.org/plink/2.0/data#set_all_var_ids,
https://samtools.github.io/bcftools/bcftools.html#norm, https://www.biostars.org/p/9531649.

**8.2 Imputed VCF (DS field) to PGEN (keeps dosages) or BED (hard calls).**
```bash
plink2 --vcf imputed.vcf.gz dosage=DS --extract-if-info 'R2 >= 0.3' --double-id \
  --max-alleles 2 --make-pgen --out geno         # lossless dosages -> use pylmm[pgen]
plink2 --pfile geno --hard-call-threshold 0.1 --make-bed --out geno_hc   # LOSSY
```
Caveats:
- BED stores hard calls only. Genotypes farther than 0.1 from an integer become **missing**, and
  dosages of exactly 0.5 or 1.5 are always missing. Expect lower power for low-INFO variants.
- Use `dosage=HDS` for Minimac4 phased dosages. `dosage=GP` reads probabilities, and plink2 errors on
  GP if a DS header is present unless you use `GP-force`.

**8.3 BGEN (for example UK Biobank) to PGEN or BED.**
```bash
plink2 --bgen ukb_imp_chr1.bgen ref-first --sample ukb.sample \
  --keep keep.txt --extract snps.txt --maf 0.01 --make-pgen --out chr1
plink2 --pfile chr1 --make-bed --out chr1_hc      # hard calls, LOSSY as in 8.2
```
Caveats:
- `ref-first` and `ref-last` must match how the BGEN was written; UK Biobank is `ref-first`. Without
  either, the last allele is treated as provisional REF.
- BGEN v1.2+ may embed sample IDs, in which case `--sample` is optional.
- Converting one chromosome at a time keeps memory bounded.

Source: https://www.cog-genomics.org/plink/2.0/input#bgen.

**8.4 PGEN to BED.**
```bash
plink2 --pfile in --max-alleles 2 --make-bed --out out          # or split: --make-pgen multiallelics=- first
```

**8.5 Controlling the allele order.**
- plink 1.9: always pass `--keep-allele-order`. Otherwise it swaps A1 to the minor allele, which flips
  pylmm's beta sign between datasets.
- plink2: `--ref-allele`/`--alt1-allele` with a file, or `--ref-from-fa --fa ref.fa` to fix REF from the
  reference genome.

**8.6 VCF Zarr (VCZ) to BED.**
```bash
vcf2zarr convert in.vcf.gz in.vcz                    # bio2zarr, if starting from VCF
vcztools view-plink --max-alleles 2 -o geno in.vcz   # BED byte-identical to plink2 --vcf --make-bed
```
Alternatively, in Python, pass `ds.call_genotype.sum(-1)` as dosages to the array API with
`ds.sample_id` and `ds.variant_contig`/`ds.variant_position`. Check `vcztools view-plink --help` for
the exact flags, since the subcommand is recent. Source: https://github.com/sgkit-dev/vcztools,
https://github.com/sgkit-dev/bio2zarr.

**8.7 Hail MatrixTable to BED.**
```python
mt = hl.split_multi_hts(mt)
hl.export_plink(mt, "geno", ind_id=mt.s, fam_id=mt.s, varid=hl.variant_str(mt.locus, mt.alleles))
```
Source: https://hail.is/docs/0.2/methods/impex.html#hail.methods.export_plink.

**8.8 Kinship from other tools.**
```bash
gcta64 --bfile geno --make-grm --out geno            # .grm.bin/.grm.N.bin/.grm.id (native read, item 4)
plink2 --bfile geno --make-grm-bin --out geno        # same GCTA format
gemma -bfile geno -gk 1 -o geno                      # output/geno.cXX.txt: plain n x n, .fam order
```
Until item 4 lands, convert GCTA to pylmm text with numpy:
```python
import numpy as np
ids = np.loadtxt("geno.grm.id", dtype=str); n = len(ids)
tri = np.fromfile("geno.grm.bin", dtype=np.float32)
K = np.zeros((n, n)); K[np.tril_indices(n)] = tri; K = K + K.T - np.diag(K.diagonal())
np.savetxt("geno.kin", K)        # rows MUST follow geno.fam order; reorder by IID if not
```
GEMMA `-gk 1` is a centred relatedness matrix. pylmm's default is standardized (the "realized
relationship matrix"). Both are valid, but h² estimates differ slightly; see bug 8.

**8.9 Phenotype and covariate files from a DataFrame.**
```python
df[["FID", "IID", "BMI"]].to_csv("pheno.txt", sep="\t", index=False, na_rep="NA")
```
Today pylmm needs no header (`header=False`). After item 2, a header is supported and preferred.

**8.10 Downstream use of pylmm output.**
```bash
uvx --python 3.13 --from gwas-sumstats-tools gwas-ssf validate out.tsv.gz -m 1   # GWAS-SSF check (after --ssf)
munge_sumstats.py --sumstats out.tsv.gz --snp rs_id --a1 effect_allele --a2 other_allele \
  --p p_value --signed-sumstats beta,0 --N-col n --merge-alleles w_hm3.snplist --out trait   # LDSC
zcat out.tsv.gz | bgzip > out.tsv.bgz && tabix -s1 -b2 -e2 -S1 out.tsv.bgz   # region queries; input must be sorted
```
LDSC needs rsIDs, so run `plink2 --set-all-var-ids` only on files that already lack rsIDs, or annotate
rsIDs (for example with MungeSumstats) first. Sources:
https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format, https://github.com/EBISPOT/gwas-sumstats-tools.

## 9. Open decisions for Nick

1. **The effect allele in the new schema.** Should the counted allele be A1/ALT (matching plink2,
   REGENIE, GWAS-SSF; the sign flips relative to today) or stay A2 (matching today)? The recommendation
   is A1, with only the legacy writer keeping A2.
2. **Beta scale.** Should the default report per-allele beta and SE (recommended; matches every other
   tool)? The standardized beta could be kept as an extra column `beta_std`.
3. **The Python floor.** Should it move to ≥ 3.11 at cutover? 3.10 reaches end of life in Oct 2026, and
   pandas 3 / pyarrow 26 need 3.11.
4. **Timing of alignment.** Should the alignment layer (item 2) land in Phase 5/5b as a robustness fix,
   rather than waiting for Phase 6?
5. **pgenlib's LGPL-3.0 license.** It is acceptable as an optional extra, but confirm.

## 10. Verification of the §8 recipes (Phase 5c, 2026-10-18)

The conversion recipes were run on a small test VCF with plink2 v2.0.0-a.7.13, bcftools 1.24, bio2zarr
0.2.1 and vcztools 0.2.0. The checked commands are in `skills/convert-genotype-formats/reference.md`.
Changes to §8:

- `plink2 --extract-if-info 'R2 >= 0.3'` is valid (alias `--extract-if`; the three-token form
  `R2 '>=' 0.3` works too).
- `vcztools view-plink` takes `-o/--output STEM` (required) and `-M/--max-alleles INT`; its `.bed` was
  byte-identical to plink2's on biallelic data.
- `plink2 --vcf ... dosage=DS` rejects multiallelic records even with `--max-alleles 2`; split them first.
- `plink2 --bgen` now requires `ref-first`, `ref-last` or `ref-unknown` (the "provisional REF" default in
  §8.3 is gone).
- §1 describes the code before Phase 5b: kinships now carry a `.id` sidecar, phenotype files may have a
  header, and individuals absent from a phenotype file are `nan` rather than skipped.
