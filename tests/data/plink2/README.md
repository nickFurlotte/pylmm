# plink2-written kinship fixtures

Small files written by **PLINK v2.0.0-a.7.13 M1 (15 Oct 2026)** (macOS arm64), so that
`tests/test_kinship_io.py` checks pylmm's GCTA GRM and `.rel` readers against an
external tool, not against files pylmm wrote itself.

`geno.bed/.bim/.fam`: 50 samples x 300 SNPs from `tests/synthetic_bed.py`
(`write_synthetic_bed(prefix, 50, 300, seed=3)`; 2% missing calls). 50 samples is the
smallest set plink2 computes allele frequencies from without `--read-freq`.

The commands (run in this directory, with plink2 on the PATH):

```text
plink2 --bfile geno --make-grm-bin --out geno                 # geno.grm.bin / .grm.N.bin / .grm.id
plink2 --bfile geno --make-rel --out geno_tri                 # text, lower triangle (the default)
plink2 --bfile geno --make-rel bin4 square --out geno_sq4     # float32 square .rel.bin
# the same samples without an FID column (.psam header "#IID"):
plink2 --bfile geno --make-pgen --out tmp
(echo "#IID"; tail -n +2 tmp.psam | cut -f2) > tmpi.psam
cp tmp.pgen tmpi.pgen; cp tmp.pvar tmpi.pvar
plink2 --pfile tmpi --make-rel triangle bin --out geno_iid    # float64 triangle; .rel.id "#IID"
```

The `.log` and temporary files were deleted. These are plink2's GRMs (frequencies from
these 50 samples, missingness-corrected), not pylmm's: they are not positive
semi-definite and differ from `pylmm kinship`'s K; the tests compare the file layouts
and IDs, not the kinship definitions.

## plink2 `--glm` reference (tests/test_sumstats.py)

`geno.pheno`: a phenotype for these 50 samples (FID IID y, with a header), written by a
seeded script: `y = 0.8 c10 - 0.5 c20 + N(0, 1)`, `c` the mean-imputed copies of A1 of
snp10 and snp20, `numpy.random.default_rng(11)`. `geno_alt.y.glm.linear` is plink2's
linear regression of it on each SNP, A1 = ALT (the `.bim` column 5):

```text
plink2 --bfile geno --pheno geno.pheno --glm allow-no-covars omit-ref --out geno_alt
```

(Without `omit-ref`, plink2 --glm takes the minor allele as A1.) pylmm gwas with an identity
kinship is the same least-squares fit, so its summary-statistics columns must match these
to plink2's printed precision (6 significant digits).
